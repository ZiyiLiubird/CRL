import numpy as np
import torch

def rollout(env,
            agent,
            index,
            adapter=None,
            testing=False,
            action_dim=None,
            hyperparam_dim=None,
            adapt_steps=200,
            max_path_length=np.inf,
            accum_context=True,
            animated=False,
            save_frames=False,
            onpolicy=True):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    current_adapts = 0

    if animated:
        env.render()
    # if testing:
    #     # agent.update_params()
    #     model_dict = agent.test_policy.state_dict()
    #     state_dict = {k:v for k,v in agent.policy.state_dict().items() if k in model_dict.keys()}
    #     model_dict.update(state_dict)
    #     agent.test_policy.load_state_dict(model_dict)
    while path_length < max_path_length:
        if testing and path_length == 0:
            idx = 0
            a, agent_info = agent.get_action(o,
                                            index=idx,
                                            testing=testing)
        else:
            idx = index
            a, agent_info = agent.get_action(o,
                                            index=idx,
                                            testing=testing)
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        # if testing, then use the adapter to predict linear weights
        if testing and current_adapts < adapt_steps:
            obs = torch.unsqueeze(torch.Tensor(o), 0)
            action = torch.unsqueeze(torch.Tensor(a), 0)
            reward = torch.unsqueeze(torch.Tensor(np.array([r])), 0)
            next_obs = torch.unsqueeze(torch.Tensor(next_o), 0)
            # if onpolicy:
            #     agent.infer_posterior(agent.context)
            # z = agent.z
            value = hyperparam_dim * action_dim
            new_weights = adapter(obs.cuda(0), action.cuda(0), reward.cuda(0), next_obs.cuda(0))

            new_weights = torch.squeeze(new_weights, 0)
            w = new_weights[:value]
            b = new_weights[value:]
            new_weights = torch.reshape(w, (action_dim, hyperparam_dim))
            #only use the 0th output layer for all testing
            #Hardcoded for now
            agent.policy.state_dict()['lc-1.weight'].copy_(new_weights)
            agent.policy.state_dict()['lc-1.bias'].copy_(b)
        
        current_adapts += 1
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]

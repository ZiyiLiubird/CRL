from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class VBFASoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            action_dim,
            hyperparam_dim,
            nets,
            adapt_steps=200,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            adapter_lr=3e-4,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            adapter=nets[4],
            action_dim=action_dim,
            hyperparam_dim=hyperparam_dim,
            # test_policy=nets[-1],
            adapt_steps=adapt_steps,
            **kwargs
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.adapter_criterion = nn.MSELoss(reduction='sum')
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf, self.adapter = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )
        self.adapter_optimizer = optimizer_class(
            self.adapter.parameters(),
            lr=adapter_lr,
        )

    # Torch stuff #
    @property
    def networks(self):
        '''
        I should reimplement it for my architecture.
        self.agent.networks including context_encoder and policy
        '''
        return self.agent.networks + [self.agent] + [self.qf1,
                                                     self.qf2,
                                                     self.vf,
                                                     self.target_vf,
                                                     self.adapter]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)  # training mode

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    # Data handling #
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements
            这里升维是因为sample_sac中会将不同任务的o,a,r...cat起来
        '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training
        the actor-critic '''
        # this batch consists of transitions sampled
        # randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(
            self.replay_buffer.random_batch(idx,
                                            batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together 所有任务batches的o为一组，a为一组，r为一组...
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of
            tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        # 每个任务一个batch
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        # 对每个任务的batch做解包
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together 将类似元素分组, o聚成一组, a聚成一组, ...
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # (o,a,r,no,t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        # print("context shape before final cat: len context:{}, inner shape{}".format(len(context), context[0].shape))
        # context shape before final cat: len context:5, inner shapetorch.Size([16, 100, 20])
        # 16 tasks, 100 is the size of batch
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # print("context shape after final cat:{}".format(context.shape))
        # context shape after final cat:torch.Size([16, 100, 47])
        return context

    ##### Training #####
    def _do_training(self, indices):

        # mb_size = self.embedding_mini_batch_size
        # num_tasks = int(len(self.train_tasks))
        num_tasks = 1
        policy_LossCombined = None
        qf_LossCombined = None
        vf_LossCombined = None
        kl_LossCombined = []
        total = 0
        adapter_LossCombined = None
        # self.agent.clear_z(num_tasks=num_tasks)

        for i in indices:
            # before: sample context batch (16, 100, 47)
            # now: sample context batch (1, 100, 47)
            context_batch = self.sample_context([i])

            # zero out context and hidden encoder state
            self.agent.clear_z(num_tasks=num_tasks)

            # context_batch before : (16, 100, 47)
            # context_batch now :    (1, 100, 47)
            # context = context_batch[:, 0:mb_size, :]
            context = context_batch
            # self._take_step([i], context)
            indice = [i]
            # data is (task, batch, feat)
            obs, actions, rewards, next_obs, terms = self.sample_sac(indice)

            # run inference in networks
            policy_outputs, task_z = self.agent(obs, indice[0], context)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

            # flattens out the task dimension, multitask learning
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            rewards_flat = rewards.view(self.batch_size * num_tasks, -1)

            # adapter networks
            # with torch.no_grad():
            # prediction = self.adapter(task_z.detach())
            # prediction = self.adapter(task_z[0].detach())
            prediction = self.adapter(obs, actions, rewards_flat, next_obs)
            weights = torch.flatten(self.agent.policy.state_dict()['lc' + str(indice[0]) + '.weight'])
            bias = torch.flatten(self.agent.policy.state_dict()['lc' + str(indice[0]) + '.bias'])
            weights_bias_concat = torch.cat((weights, bias))
            adapter_loss = self.adapter_criterion(prediction, weights_bias_concat)

            if adapter_LossCombined is None:
                adapter_LossCombined = adapter_loss
            else:
                adapter_LossCombined = adapter_LossCombined + adapter_loss
            # Q and V networks
            # encoder will only get gradients from Q nets
            q1_pred = self.qf1(indice[0], obs, actions, task_z)
            q2_pred = self.qf2(indice[0], obs, actions, task_z)
            v_pred = self.vf(indice[0], obs, task_z.detach())
            # get targets for use in V and Q updates
            with torch.no_grad():
                target_v_values = self.target_vf(indice[0], next_obs, task_z.detach())

            # KL constraint on z if probabilistic
            # self.context_optimizer.zero_grad()
            if self.use_information_bottleneck:
                kl_div = self.agent.compute_kl_div()
                kl_LossCombined.extend(kl_div)
                # kl_loss = self.kl_lambda * kl_div

            # qf and encoder update (note encoder does not get grads from policy or vf)
            # self.qf1_optimizer.zero_grad()
            # self.qf2_optimizer.zero_grad()
            # rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            # scale rewards for Bellman update
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = terms.view(self.batch_size * num_tasks, -1)
            q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
            qf_loss = qf1_loss + qf2_loss
            # qf_loss.backward()
            # self.qf1_optimizer.step()
            # self.qf2_optimizer.step()
            # self.context_optimizer.step()

            if qf_LossCombined is None:
                qf_LossCombined = qf_loss
            else:
                qf_LossCombined = qf_LossCombined + qf_loss

            # compute min Q on the new actions
            min_q_new_actions = self._min_q(indice[0], obs, new_actions, task_z)

            # vf update
            v_target = min_q_new_actions - log_pi
            vf_loss = self.vf_criterion(v_pred, v_target.detach())
            if vf_LossCombined is None:
                vf_LossCombined = vf_loss
            else:
                vf_LossCombined = vf_LossCombined + vf_loss

            # policy update
            # n.b. policy update includes dQ/da
            log_policy_target = min_q_new_actions

            policy_loss = (
                    log_pi - log_policy_target
            ).mean()

            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2)
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            # std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2)
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            # pre_activation_reg_loss = self.policy_pre_activation_weight * (
            #     (pre_tanh_value**2).sum(dim=1)
            # )
            policy_loss = policy_loss[None, ...]
            mean_reg_loss = mean_reg_loss[None, ...]
            std_reg_loss = std_reg_loss[None, ...]
            pre_activation_reg_loss = pre_activation_reg_loss[None, ...]
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss
            # print("policy_loss:{}".format(policy_loss)) # nan
            if policy_LossCombined is None:
                policy_LossCombined = policy_loss
            else:
                policy_LossCombined = torch.cat((policy_LossCombined, policy_loss), dim=0)

            total += 1

        #Mean loss over all training tasks
        policy_LossCombined = policy_LossCombined.mean()
        qf_LossCombined = qf_LossCombined / total
        vf_LossCombined = vf_LossCombined / total
        kl_div = torch.sum(torch.stack(kl_LossCombined))
        kl_loss = self.kl_lambda * kl_div

        self.adapter_optimizer.zero_grad()
        adapter_LossCombined.backward(retain_graph=True)
        self.adapter_optimizer.step()

        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_loss.backward(retain_graph=True)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_LossCombined.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()
        
        self.vf_optimizer.zero_grad()
        vf_LossCombined.backward(retain_graph=True)
        self.vf_optimizer.step()


        self.policy_optimizer.zero_grad()
        policy_LossCombined.backward(retain_graph=True)
        self.policy_optimizer.step()


        self._update_target_network()
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['adapter Loss'] = ptu.get_numpy(adapter_LossCombined)
            self.eval_statistics['QF Loss'] = ptu.get_numpy(qf_LossCombined)
            self.eval_statistics['VF Loss'] = ptu.get_numpy(vf_LossCombined)
            self.eval_statistics['Policy Loss'] = ptu.get_numpy(
                policy_LossCombined
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

        self.agent.detach_z()
    def _min_q(self, index, obs, actions, task_z):
        '''
        min_q的更新不要更新z的梯度
        '''
        q1 = self.qf1(index, obs, actions, task_z.detach())
        q2 = self.qf2(index, obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    # def _take_step(self, indices, context):

        # num_tasks = len(indices)
        # policy_LossCombined = None
        # qf_LossCombined = None
        # vf_LossCombined = None
        # total = None
        # adapter_LossCombined = None

        # # data is (task, batch, feat)
        # obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # # run inference in networks
        # policy_outputs, task_z = self.agent(obs, indices[0], context)
        # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # # flattens out the task dimension, multitask learning
        # t, b, _ = obs.size()
        # obs = obs.view(t * b, -1)
        # actions = actions.view(t * b, -1)
        # next_obs = next_obs.view(t * b, -1)

        # # adapter networks
        # prediction = self.adapter(task_z.detach())
        # weights = torch.flatten(self.agent.policy.state_dict()['lc' + str(indices[0]) + '.weight'])
        # bias = torch.flatten(self.agent.policy.state_dict()['lc' + str(indices[0]) + '.bias'])
        # weights_bias_concat = torch.cat((weights, bias))
        # # print("adapter prediction shape:{}".format(prediction.shape))
        # # print("adapter label shape:{}".format(weights_bias_concat.shape))
        # adapter_loss = self.adapter_criterion(prediction, weights_bias_concat)

        # if adapter_LossCombined is None:
        #     adapter_LossCombined = adapter_loss
        # else:
        #     adapter_LossCombined = adapter_LossCombined + adapter_loss
        # # Q and V networks
        # # encoder will only get gradients from Q nets
        # q1_pred = self.qf1(indices[0], obs, actions, task_z)
        # q2_pred = self.qf2(indices[0], obs, actions, task_z)
        # v_pred = self.vf(indices[0], obs, task_z.detach())
        # # get targets for use in V and Q updates
        # with torch.no_grad():
        #     target_v_values = self.target_vf(indices[0], next_obs, task_z)

        # # KL constraint on z if probabilistic
        # self.context_optimizer.zero_grad()
        # if self.use_information_bottleneck:
        #     kl_div = self.agent.compute_kl_div()
        #     kl_loss = self.kl_lambda * kl_div
        #     kl_loss.backward(retain_graph=True)

        # # qf and encoder update (note encoder does not get grads from policy or vf)
        # # self.qf1_optimizer.zero_grad()
        # # self.qf2_optimizer.zero_grad()
        # rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # # scale rewards for Bellman update
        # rewards_flat = rewards_flat * self.reward_scale
        # terms_flat = terms.view(self.batch_size * num_tasks, -1)
        # q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        # qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        # # qf_loss.backward()
        # # self.qf1_optimizer.step()
        # # self.qf2_optimizer.step()
        # self.context_optimizer.step()

        # # compute min Q on the new actions
        # min_q_new_actions = self._min_q(indices[0], obs, new_actions, task_z)

        # # vf update
        # v_target = min_q_new_actions - log_pi
        # vf_loss = self.vf_criterion(v_pred, v_target.detach())
        # # self.vf_optimizer.zero_grad()
        # # vf_loss.backward()
        # # self.vf_optimizer.step()
        # # self._update_target_network()

        # # policy update
        # # n.b. policy update includes dQ/da
        # log_policy_target = min_q_new_actions

        # policy_loss = (
        #         log_pi - log_policy_target
        # ).mean()

        # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        # std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        # pre_tanh_value = policy_outputs[-1]
        # pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #     (pre_tanh_value**2).sum(dim=1).mean()
        # )
        # policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        # policy_loss = policy_loss + policy_reg_loss
        # if policy_LossCombined is None:
        #     policy_LossCombined = policy_loss
        # else:
        #     policy_LossCombined = torch.cat((policy_LossCombined, policy_loss), dim=0)

        # if total is None:
        #     total = q1_pred.shape[0]
        # else:
        #     total += q1_pred.shape[0]

        # if qf_LossCombined is None:
        #     qf_LossCombined = qf_loss
        # else:
        #     qf_LossCombined = qf_LossCombined + qf_loss

        # if vf_LossCombined is None:
        #     vf_LossCombined = vf_loss
        # else:
        #     vf_LossCombined = vf_LossCombined + vf_loss

        # # self.policy_optimizer.zero_grad()
        # # policy_loss.backward()
        # # self.policy_optimizer.step()

        # # policy_LossCombined = policy_LossCombined.mean()
        # # qf_LossCombined = qf_LossCombined / total
        # # vf_LossCombined = vf_LossCombined / total

        # # save some statistics for eval
        # if self.eval_statistics is None:
        #     # eval should set this to None.
        #     # this way, these statistics are only computed for one batch.
        #     self.eval_statistics = OrderedDict()
        #     if self.use_information_bottleneck:
        #         z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
        #         z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
        #         self.eval_statistics['Z mean train'] = z_mean
        #         self.eval_statistics['Z variance train'] = z_sig
        #         self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
        #         self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

        #     self.eval_statistics['QF Loss'] = ptu.get_numpy(qf_LossCombined)
        #     self.eval_statistics['VF Loss'] = ptu.get_numpy(vf_LossCombined)
        #     self.eval_statistics['Policy Loss'] = ptu.get_numpy(
        #         policy_LossCombined
        #     )
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Q Predictions',
        #         ptu.get_numpy(q1_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'V Predictions',
        #         ptu.get_numpy(v_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Log Pis',
        #         ptu.get_numpy(log_pi),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy mu',
        #         ptu.get_numpy(policy_mean),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy log std',
        #         ptu.get_numpy(policy_log_std),
        #     ))
        # #Mean loss over all training tasks
        
        # self.adapter_optimizer.zero_grad()
        # adapter_LossCombined.backward()
        # self.adapter_optimizer.step()
        
        # self.policy_optimizer.zero_grad()
        # policy_LossCombined.backward(retain_graph=True)
        # self.policy_optimizer.step()
        
        # self.qf1_optimizer.zero_grad()
        # self.qf2_optimizer.zero_grad()
        # qf_LossCombined.backward(retain_graph=True)
        # self.qf1_optimizer.step()
        # self.qf2_optimizer.step()
        
        # self.vf_optimizer.zero_grad()
        # vf_LossCombined.backward(retain_graph=True)
        # self.vf_optimizer.step()
        
        # self._update_target_network()


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            adapter=self.adapter.state_dict(),
        )
        return snapshot

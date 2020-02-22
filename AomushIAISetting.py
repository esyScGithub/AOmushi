import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import AomushI as ai
import numpy as np
import json
import AomushIML
import itertools as it
import os
import datetime
import pathlib

TERGET_DIR_ROOT = 'E:/Git/AomushI/Result/Set_20200222_1109/test_20200222_110938/'
TERGET_DIR_AGENT = TERGET_DIR_ROOT + 'lastAgent'


def AomushILarningMain():
    aomushiEnv = ai.SnakeGameCore()
    # aomushiEnv.run()

    # 環境を初期化（戻り値で、初期状態の観測データobservationが取得できる）
    obs = aomushiEnv.reset()

    # list_adam_eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # list_gamma = [1-i*0.01 for i in range(1,8)]
    list_adam_eps = [1e-2]
    list_gamma = [0.99]

    '''
    各パラメータの意味
        ■Adam
        adam_eps: 
        start_epsilon: 
        end_epsilon:
        decay_steps:
        hidden_layer:
        hidden_nodes:
        kaseika_func:
        gamma:
        replay_start_size:
        update_interval:
        target_update_interval:
        ER_capacity:
        n_episodes:
        max_episode_len:
    '''
    
    # ベストスコア記録用変数の定義
    bestReward = -9999
    bestAverageReward = -9999
    bestDirPath = ''
    bestAveDirPath = ''


    # 検証セットのディレクトリ作成
    dirPath = os.path.dirname(__file__) + '/Result/Set_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    pathlib.Path(dirPath).mkdir()

    # 検証する塊単位でデータを記録しておく
    f = open(dirPath + '/TotalSummary.txt', 'w')
    f.write(f'Range of Parameter\n')
    f.write(f'{list_adam_eps=}' + '\n')
    f.write(f'{list_gamma=}' + '\n')
    f.write('\n')

    # 変動させるパラメータの全通りの組み合わせを実行
    try:
        for tempAdam, tempGamma in it.product(list_adam_eps, list_gamma):
            # パラメータ設定
            paramDic = {'adam_eps': tempAdam,
                        'start_epsilon': 0.5,
                        'end_epsilon': 0.3,
                        'decay_steps': 500,
                        'hidden_layer': 3,
                        'hidden_nodes': 200,
                        'kaseika_func': 'relu',
                        'gamma': tempGamma,
                        'replay_start_size': 500,
                        'update_interval': 1,
                        'target_update_interval': 100,
                        'ER_capacity': 10 ** 4 * 2,
                        'n_episodes': 50000,
                        'max_episode_len': 50000
                        }

            class QFunction(chainer.Chain):

                def __init__(self, obs_size, n_actions, n_hidden_channels=paramDic['hidden_nodes']):
                    super().__init__()
                    with self.init_scope():
                        # 各層の数を定義している。以下の場合は4レイヤー構造。
                        self.l0 = L.Linear(obs_size, n_hidden_channels)
                        self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
                        self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
                        self.l3 = L.Linear(n_hidden_channels, n_actions)

                def __call__(self, x, test=False):
                    """
                    Args:
                        x (ndarray or chainer.Variable): An observation
                        test (bool): a flag indicating whether it is in test mode
                    """
                    h = F.relu(self.l0(x))
                    h = F.relu(self.l1(h))
                    h = F.relu(self.l2(h))
                    return chainerrl.action_value.DiscreteActionValue(self.l3(h))

            # 環境サイズとアクションのサイズを取得して、InputとOutputのノード数を決める。
            obs_size = obs.reshape(1,-1).shape[1]
            n_actions = 4

            # Q関数の定義
            q_func = QFunction(obs_size, n_actions)

            # CUDAの使用。（使用する場合はコメントを外す）
            # q_func.to_gpu(0)

            # オプティマイザーの設定
            optimizer = chainer.optimizers.Adam(eps=paramDic['adam_eps'])
            optimizer.setup(q_func)

            # Epsilon-Greedyの設定（初期値から線形的に減少させる）
            explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=paramDic['start_epsilon'], end_epsilon=paramDic['end_epsilon'], decay_steps=paramDic['decay_steps'], random_action_func=aomushiEnv.actionSample)

            # DQN uses Experience Replay.
            # Specify a replay buffer and its capacity.
            replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=paramDic['ER_capacity'])

            # Since observations from CartPole-v0 is numpy.float64 while
            # Chainer only accepts numpy.float32 by default, specify
            # a converter as a feature extractor function phi.
            phi = lambda x: x.astype(np.float32, copy=False)

            # Now create an agent that will interact with the envEnvironment.
            agent = chainerrl.agents.DoubleDQN(
                q_func, optimizer, replay_buffer, paramDic['gamma'], explorer,
                replay_start_size=paramDic['replay_start_size'], update_interval=paramDic['update_interval'],
                target_update_interval=paramDic['target_update_interval'], phi=phi)

            # 学習アルゴリズムを実行
            reward, averageReward, tempDirPath = AomushIML.AomushILearning(agent, paramDic, dirPath)

            if reward > bestReward:
                bestReward = reward
                bestParam = {'adam_eps': tempAdam, 'gamma': tempGamma}
                bestDirPath = tempDirPath

            if averageReward > bestAverageReward:
                bestAverageReward = averageReward
                bestAberageRewardParam = {'adam_eps': tempAdam, 'gamma': tempGamma}
                bestAveDirPath = tempDirPath

    except Exception as e:
        f.wirte(f'{e=}')

    finally:
        f.write(f'MaxParameter\n')
        f.write(f'{bestReward=}' + '\n')
        f.write(f'{bestParam=}' + '\n')
        f.write(f'{bestDirPath=}' + '\n')
        f.write('\n')
        f.write(f'BestAverageParameter\n')
        f.write(f'{bestAverageReward=}' + '\n')
        f.write(f'{bestAberageRewardParam=}' + '\n')
        f.write(f'{bestAveDirPath=}' + '\n')
        f.close

# ゲーム本体用に残しているが、上と統合できるなら消して良い。
def AomushiModelRead(newModel=False):
    aomushiEnv = ai.SnakeGameCore()
    # aomushiEnv.run()

    # 環境を初期化（戻り値で、初期状態の観測データobservationが取得できる）
    obs = aomushiEnv.reset()

    # パラメータ設定
    if newModel==False:
        with open(TERGET_DIR_ROOT + 'param.json', 'r') as f:
            paramDic = json.load(f)
    else:
        '''
        各パラメータの意味
          ■Adam
            adam_eps: 
            start_epsilon: 
            end_epsilon:
            decay_steps:
            hidden_layer:
            hidden_nodes:
            kaseika_func:
            gamma:
            replay_start_size:
            update_interval:
            target_update_interval:
            ER_capacity:
            n_episodes:
            max_episode_len:
        '''
        paramDic = {'adam_eps': 1e-3,
                    'start_epsilon': 0.3,
                    'end_epsilon': 0.1,
                    'decay_steps': 500,
                    'hidden_layer': 3,
                    'hidden_nodes': 200,
                    'kaseika_func': 'relu',
                    'gamma': 0.95,
                    'replay_start_size': 500,
                    'update_interval': 1,
                    'target_update_interval': 100,
                    'ER_capacity': 10 ** 4 * 2,
                    'n_episodes': 2000,
                    'max_episode_len': 50000
                    }

    class QFunction(chainer.Chain):

        def __init__(self, obs_size, n_actions, n_hidden_channels=paramDic['hidden_nodes']):
            super().__init__()
            with self.init_scope():
                # 各層の数を定義している。以下の場合は4レイヤー構造。
                self.l0 = L.Linear(obs_size, n_hidden_channels)
                self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
                self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
                self.l3 = L.Linear(n_hidden_channels, n_actions)

        def __call__(self, x, test=False):
            """
            Args:
                x (ndarray or chainer.Variable): An observation
                test (bool): a flag indicating whether it is in test mode
            """
            h = F.relu(self.l0(x))
            h = F.relu(self.l1(h))
            h = F.relu(self.l2(h))
            return chainerrl.action_value.DiscreteActionValue(self.l3(h))

    # 環境サイズとアクションのサイズを取得して、InputとOutputのノード数を決める。
    obs_size = obs.reshape(1,-1).shape[1]
    n_actions = 4

    # Q関数の定義
    q_func = QFunction(obs_size, n_actions)

    # CUDAの使用。（使用する場合はコメントを外す）
    # q_func.to_gpu(0)

    # オプティマイザーの設定
    optimizer = chainer.optimizers.Adam(eps=paramDic['adam_eps'])
    optimizer.setup(q_func)

    # Epsilon-Greedyの設定（初期値から線形的に減少させる）
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=paramDic['start_epsilon'], end_epsilon=paramDic['end_epsilon'], decay_steps=paramDic['decay_steps'], random_action_func=aomushiEnv.actionSample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=paramDic['ER_capacity'])

    # Since observations from CartPole-v0 is numpy.float64 while
    # Chainer only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the envEnvironment.
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, paramDic['gamma'], explorer,
        replay_start_size=paramDic['replay_start_size'], update_interval=paramDic['update_interval'],
        target_update_interval=paramDic['target_update_interval'], phi=phi)

    if newModel == False:
        agent.load(TERGET_DIR_AGENT)

    return agent, paramDic

if __name__ == "__main__":
    AomushILarningMain()
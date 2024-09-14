import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class FastDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = np.zeros((10000, state_size * 2 + 3))  # [state, action, reward, next_state, done]
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory_counter = 0
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        index = self.memory_counter % 10000
        self.memory[index] = np.concatenate([state, [action, reward], next_state, [done]])
        self.memory_counter += 1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if self.memory_counter < self.batch_size:
            return
        indices = np.random.choice(min(self.memory_counter, 10000), self.batch_size, replace=False)
        batch = self.memory[indices]
        
        states = batch[:, :self.state_size]
        actions = batch[:, self.state_size].astype(int)
        rewards = batch[:, self.state_size + 1]
        next_states = batch[:, self.state_size + 2:-1]
        dones = batch[:, -1]

        targets = rewards + self.gamma * (1 - dones) * np.max(self.model.predict(next_states, verbose=0), axis=1)
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(self.batch_size), actions] = targets

        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=self.batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_fast_dqn(episodes, render=False):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = FastDQNAgent(state_size, action_size)
    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        max_steps = 500
        for time in range(max_steps):
            if render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state[0], action, reward, next_state[0], done)
            state = next_state
            score += 1
            if done:
                break
            agent.replay()
        
        scores.append(score)
        if e % 10 == 0:
            print(f"回合: {e}/{episodes}, 得分: {score}, epsilon: {agent.epsilon:.2f}")

    return scores, agent

# 訓練 DQN
episodes = 500
dqn_scores, trained_agent = train_fast_dqn(episodes)

print(f"得分數量: {len(dqn_scores)}")

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.plot(range(len(dqn_scores)), dqn_scores)
plt.title('Fast DQN 學習曲線')
plt.xlabel('回合')
plt.ylabel('得分')
plt.show()

# 評估訓練後的 DQN
env = gym.make('CartPole-v1')
trained_agent.epsilon = 0  # 使用純貪婪策略進行評估

evaluation_scores = []
for _ in range(100):  # 評估 100 次
    state, _ = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    done = False
    while not done:
        action = trained_agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        state = next_state
    evaluation_scores.append(total_reward)

print(f"Fast DQN 平均得分: {np.mean(evaluation_scores):.2f}")
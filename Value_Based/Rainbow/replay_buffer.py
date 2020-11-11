''' PER Replaybuffer '''

import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree # This is baseline provided in OpenAI.

# Naive ReplayBuffer
class ReplayBuffer:
    """ Experience Replay Buffer which is implemented in DQN paper. https://www.nature.com/articles/nature14236 
    The detailed parameter is described in each method.
    """

    def __init__(self, 
                 buffer_size: ('int: total size of the Replay Buffer'), 
                 input_dim: ('tuple: a dimension of input data. Ex) (3, 84, 84)'), 
                 batch_size: ('int: a batch size when updating')):
                 
        assert len(input_dim)==3, "The state dimension should be 3-dim! (CHxWxH). Please check if input_dim is right"

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.save_count, self.current_size = 0, 0

        self.state_buffer = np.ones((buffer_size, input_dim[0], input_dim[1], input_dim[2]), 
                                    dtype=np.uint8) # WARN: data type is np.int8 so that it should be stored ONLY in a range with 0~255 integer!!!
        self.action_buffer = np.ones(buffer_size, dtype=np.uint8) 
        self.reward_buffer = np.ones(buffer_size, dtype=np.float32) 
        self.next_state_buffer = np.ones((buffer_size, input_dim[0], input_dim[1], input_dim[2]),  
                                         dtype=np.uint8) # WARN: data type is np.int8 so that it should be stored ONLY in a range with 0~255 integer!!!
        self.done_buffer = np.ones(buffer_size, dtype=np.uint8) 

    def store(self, 
              state: np.ndarray, 
              action: int, 
              reward: float, 
              next_state: np.ndarray, 
              done: int):

        self.state_buffer[self.save_count] = state
        self.action_buffer[self.save_count] = action
        self.reward_buffer[self.save_count] = reward
        self.next_state_buffer[self.save_count] = next_state
        self.done_buffer[self.save_count] = done

        self.save_count = (self.save_count + 1) % self.buffer_size
        self.current_size = min(self.current_size+1, self.buffer_size)

    def batch_load(self):
        indices = np.random.randint(self.current_size, size=self.batch_size)
        return dict(
                states=self.state_buffer[indices], 
                actions=self.action_buffer[indices],
                rewards=self.reward_buffer[indices],
                next_states=self.next_state_buffer[indices], 
                dones=self.done_buffer[indices]) 

# ReplayBuffer for Prioritized Experience Replay. 
class PrioritizedReplayBuffer(ReplayBuffer):
    
    def __init__(self, buffer_size, input_dim, batch_size, alpha):
        
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, input_dim, batch_size)
        
        # For PER. Parameter settings. 
        self.max_priority, self.tree_idx = 1.0, 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, 
              state: np.ndarray, 
              action: int, 
              reward: float, 
              next_state: np.ndarray, 
              done: int):
        
        super().store(state, action, reward, next_state, done)
        
        self.sum_tree[self.tree_idx] = self.max_priority ** self.alpha # log N의 계산복잡도. (일반 array는 계산복잡도=0)
        self.min_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size

    def batch_load(self, beta):
        
        # index를 batch 단위로 받아오는 부분. 충분히 병렬로 만들 수 있다.
        indices, p_total = self._sample_indices_with_priority()
        weights = self._cal_weight(indices, p_total, self.current_size, beta)
        return dict(
                states=self.state_buffer[indices], 
                actions=self.action_buffer[indices],
                rewards=self.reward_buffer[indices],
                next_states=self.next_state_buffer[indices], 
                dones=self.done_buffer[indices],
                weights=weights,
                indices=indices) 

    def update_priorities(self, indices, priorities):
        # 이 부분도 병렬 처리 할 수 있는 구간.
        # print(priorities)
        for idx, priority in zip(indices, priorities):
            
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            # priority = R + Q_target - Q
            self.max_priority = max(self.max_priority, priority)
    
    def _sample_indices_with_priority(self):
        ''' SumTree에서 Leaf Node의 Value(=priority)에 비례해서 뽑는 메소드'''
        p_total = self.sum_tree.sum() 
        segment = p_total / self.batch_size
        segment_list = [i*segment for i in range(self.batch_size)]
        samples = [np.random.uniform(a, a+segment) for a in segment_list]
        indices = [self.sum_tree.find_prefixsum_idx(sample) for sample in samples]
        
        return indices, p_total
    
    def _cal_weight(self, indices, p_total, N, beta):
        
        p_min = self.min_tree.min() / p_total
        max_weight = (p_min*N) ** (-beta) # (1/(p*N))^(beta)
        
        # indices로 sampling을 하는 부분. 충분히 병렬로 만들 수 있다.
        p_samples = np.array([self.sum_tree[idx] for idx in indices]) / p_total
        weights = (p_samples*N)**(-beta)/max_weight
        return weights
    
if __name__=='__main__':
    buffer_size = 100
    state_dim = (4, 84, 84)
    batch_size = 16
    alpha = 0.6
    beta = 0.4
    buffer = PrioritizedReplayBuffer(buffer_size, state_dim, batch_size, alpha)
    for i in range(50):
        state = np.ones(state_dim)
        action = 1
        reward = 1
        next_state = np.ones(state_dim)
        done = 1
        buffer.store(state, action, reward, next_state, done)
    print(buffer.alpha)
    print(buffer.max_priority)
    print(buffer.batch_load(beta)['states'].shape)
    print(buffer.batch_load(beta)['actions'].shape)
    print(buffer.batch_load(beta)['rewards'].shape)
    print(buffer.batch_load(beta)['next_states'].shape)
    print(buffer.batch_load(beta)['dones'].shape)
    print(buffer.batch_load(beta)['weights'].shape)
    print(buffer.batch_load(beta)['indices'].__len__())
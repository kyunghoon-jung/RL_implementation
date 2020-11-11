import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree

# Naive ReplayBuffer
class ReplayBuffer:

    def __init__(self, 
                 buffer_size: int, 
                 input_dim: tuple, 
                 batch_size: int,
                 input_type: str):
        
        if input_type=='3-dim':
            assert len(input_dim)==3, "The state dimension should be 3-dim! (Channel x Width x Height). Please check if input_dim is right"

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.save_count, self.current_size = 0, 0

        if input_type=='1-dim':
            self.state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32) 
            self.action_buffer = np.ones(buffer_size, dtype=np.uint8) 
            self.reward_buffer = np.ones(buffer_size, dtype=np.float32) 
            self.next_state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32)
            self.done_buffer = np.ones(buffer_size, dtype=np.uint8) 
        else:
            self.state_buffer = np.ones((buffer_size, input_dim[0], input_dim[1], input_dim[2]), 
                                        dtype=np.uint8) # WARN: data type is np.int8 so that it should be stored ONLY 0~255 integer!!!
            self.action_buffer = np.ones(buffer_size, dtype=np.uint8) 
            self.reward_buffer = np.ones(buffer_size, dtype=np.float32) 
            self.next_state_buffer = np.ones((buffer_size, input_dim[0], input_dim[1], input_dim[2]),  
                                            dtype=np.uint8) # WARN: data type is np.int8 so that it should be stored ONLY 0~255 integer!!!
            self.done_buffer = np.ones(buffer_size, dtype=np.uint8) 
            
    def __len__(self):
        return self.current_size

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
    
    def __init__(self, buffer_size, input_dim, batch_size, alpha, input_type):
        
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, input_dim, batch_size, input_type)
        
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
        
        self.sum_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.min_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size
        
    def batch_load(self, beta):
        
        # indices를 받아오는 부분은 병렬처리!!, 그리고 같은 함수에서 weight도 받을 수 있다.
        indices = self._load_batch_indices()
        
        weights = np.array([self._calculate_weight(idx, beta) for idx in indices])
        
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
        for idx, priority in zip(indices, priorities):
            
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            
            self.max_priority = max(self.max_priority, priority)
    
    def _load_batch_indices(self):
        
        indices = []
        p_total = self.sum_tree.sum(0, len(self)-1) 
        segment = p_total / self.batch_size
        
        # multiprocessing 등을 활용해서 병렬처리 하자
        for i in range(self.batch_size):
            a = segment * i 
            b = segment * (i+1)
            sample = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(sample) # sample의 tree에서의 idx를 리턴
            indices.append(idx)

        return indices
    
    def _calculate_weight(self, idx, beta):
        
        # 이 부분은 batch 당 weight 구할 때 한번만 하면 될듯.
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min*len(self)) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample*len(self)) ** (-beta)
        weight = weight / max_weight
        return weight
    
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
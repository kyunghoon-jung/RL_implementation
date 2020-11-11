
import numpy as np

class ReplayBuffer:
    def __init__(self, 
                 buffer_size: ('int: total size of the Replay Buffer'), 
                 input_dim: ('tuple: a dimension of input data. Ex) (3, 84, 84)'), 
                 batch_size: ('int: a batch size when updating')):
                 
        assert len(input_dim)==3, "The state dimension should be 3-dim! (CHxWxH). Please check if input_dim is right"

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.save_idx, self.current_size = 0, 0

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

        self.state_buffer[self.save_idx] = state
        self.action_buffer[self.save_idx] = action
        self.reward_buffer[self.save_idx] = reward
        self.next_state_buffer[self.save_idx] = next_state
        self.done_buffer[self.save_idx] = done

        self.save_idx = (self.save_idx + 1) % self.buffer_size
        self.current_size = min(self.current_size+1, self.buffer_size)

    def batch_load(self):
        indices = np.random.randint(self.current_size, size=self.batch_size)
        return dict(
                states=self.state_buffer[indices], 
                actions=self.action_buffer[indices],
                rewards=self.reward_buffer[indices],
                next_states=self.next_state_buffer[indices], 
                dones=self.done_buffer[indices]) 
    
if __name__=='__main__':
    buffer_size = 100
    state_dim = (4, 84, 84)
    batch_size = 64
    buffer = ReplayBuffer(buffer_size, state_dim, batch_size)
    samples_s = np.ones((1000, 4, 84, 84))
    samples_a = np.ones((1000, 1))
    samples_r = np.ones((1000, 1))
    samples_n_s = np.ones((1000, 4, 84, 84))
    samples_d = np.ones((1000, 1))
    for s, a, r, n_s, d in zip(samples_s, samples_a, samples_r, samples_n_s, samples_d):
        buffer.store(s, a, r, n_s, d)

    print(buffer.batch_load()['states'].shape)
    print(buffer.batch_load()['rewards'].shape)
    print(buffer.batch_load()['dones'].shape)
    print(buffer.batch_load()['next_states'].shape)
    print(buffer.batch_load()['actions'].shape)
    print(buffer.current_size)
    print(buffer.batch_size)
    
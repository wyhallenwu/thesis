# import os
# from queue import Queue


# class ClientBuffer():
#     def __init__(self, dataset_path, buffer_size=5):
#         self.dataset_path = dataset_path
#         self.current_frame = 1
#         self.buffer_size = buffer_size
#         self.buffer = Queue(buffer_size)

#     def __len__(self):
#         return len(self.buffer)

#     def full(self):
#         return self.buffer.full()

#     def empty(self):
#         return self.buffer.empty()

#     def get_chunk_frames_id(self):
#         return None if self.buffer.empty() else self.buffer.get()

#     def qsize(self):
#         return self.buffer.qsize()

#     def get_frame_path(self, frame_index):
#         return self.dataset_path + "/" + self.transmission_set[frame_index - 1]

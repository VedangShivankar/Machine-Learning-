a = ["hey", "bro", "your", "awesome"]

itr = reversed(a)
print(next(itr))
print(next(itr))

#section 2

class RemoteControl():
    def __init__(self):
        self.channels = ["HBO", "cnn", "abc", "espn"]
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index == len(self.channels):
            raise StopIteration

        return self.channels[self.index]

r = RemoteControl()

itr = iter(r)
print(next(itr))
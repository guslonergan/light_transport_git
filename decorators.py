import numpy as np
array = np.ndarray
import random

# def wrap(f):
#     def wrapped_f(*args):
#         ego = args[0]
#         args = tuple(number(arg.number*ego.m,ego.m) for arg in args)
#         return f(*args)
#     return wrapped_f


# class number:
#     def __init__(self,number,m):
#         self.number = number
#         self.m = m

#     @wrap
#     def add(self,other):
#         return number(self.number + other.number, self.m)

# x = number(10,2)
# y = number(-5,3)
# print(y.add(x).number)


# ------------------------------------------------------------


# def wrap(f):
#     def wrapped_f(*args):
#         ego = args[0]
#         return ego.output_transform(f(*ego.input_transform(*args)))
#     return wrapped_f


# class number:
#     def __init__(self,number,m):
#         self.number = number
#         self.m = m

#     def input_transform(self,*input):
#         return tuple(number(self.m*entry.number,self.m) for entry in input)

#     def output_transform(self,output):
#         return number(self.m + output.number,self.m)

#     @wrap
#     def add(self,other):
#         return number(self.number + other.number, self.m)


# x = number(10,2)
# y = number(-5,3)
# print(y.add(x).number)



# ------------------------------------------------------------


# def wrap(f):
#     def wrapped_f(*args):
#         selfie = args[0]
#         return f(*selfie.input_transform(*args))
#     return wrapped_f

# class framed_vector:
#     def __init__(self, vector, frame):
#         self._vector = vector
#         self.array = array(vector)
#         self.frame = frame

#     def input_transform(self, *inputs):
#         return tuple(np.dot(self.frame, entry) for entry in inputs)

#     # @wrap
#     # def dot(self, other):
#         # self.array.dot(other)

#     def dot(self, other):
#         self.array.dot(*self.input_transform(self.array, other.array))


# mat = np.array([[1,2],[3,4]])
# v = np.array([1,0])
# v = framed_vector(v,mat)

# print(v.dot(v))

# ------------------------------------------------------------

# class x:
#     def __init__(self):
#         self.r = self.get()

#     def get(self):
#         return random.uniform(0,1)

# x1 = x()
# print(x1.r)
# print(x1.r)

# ------------------------------------------------------------

# class thing:
#     def __init__(self,a,b):
#         self.a = a
#         self.b = b
#         self.d = a + b

#     @staticmethod
#     def initializers():
#         return ('a','b')

#     @staticmethod
#     def create(other):
#         return thing(*tuple(getattr(other,initializer) for initializer in thing.initializers()))


# class thingy(thing):
#     def __init__(self,a,b,c):
#         self.c = c
#         super().__init__(a,b)

#     @staticmethod
#     def initializers():
#         return ('a','b','c')

#     @staticmethod
#     def create(other):
#         return thing(*tuple(getattr(other,initializer) for initializer in thing.initializers()))

#     def regress(self):
#         return super().create(self)

# x = thingy(1,2,4)

# y = x.regress()

# print(y.d)

# ------------------------------------------------------------

x = [1]
x.reverse()
print(x)























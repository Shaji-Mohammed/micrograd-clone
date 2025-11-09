"""
    The out put from draw_dot() can pe pasted here for the viz:

    https://dreampuf.github.io/GraphvizOnline/?engine=dot#digraph%20%7B%0A%09graph%20%5Brankdir%3DLR%5D%0A%0943208200%20%5Blabel%3D%22%7B%20b%20%7C%20data%205.0000%20%7C%20grad%200.0000%20%7D%22%20shape%3Drecord%5D%0A%0941136072%20%5Blabel%3D%22%7B%20a%20%7C%20data%204.0000%20%7C%20grad%200.0000%20%7D%22%20shape%3Drecord%5D%0A%0943635696%20%5Blabel%3D%22%7B%20c%20%7C%20data%209.0000%20%7C%20grad%200.0000%20%7D%22%20shape%3Drecord%5D%0A%09%2243635696%2B%22%20%5Blabel%3D%22%2B%22%5D%0A%09%2243635696%2B%22%20-%3E%2043635696%0A%0943208200%20-%3E%20%2243635696%2B%22%0A%0941136072%20-%3E%20%2243635696%2B%22%0A%7D%0A
"""

# Ex. 1 -----
a = Value(4.0, label='a')
b = Value(5.0, label='b')
c = a + b; c.label = 'c'
print(draw_dot(c))


# Ex. 2 ----

# inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# bais of the neuron
b = Value(6.7, label="b")

# 
x1w1 = x1*w1; x1w1.label="x1*w1"
x2w2 = x2*w2; x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label="x1w1 + x2w2"
n = x1w1x2w2 + b; n.label="n"
o = n.tanh(); o.label = 'o'
print(draw_dot(o))
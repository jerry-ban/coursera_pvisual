import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.interactive(True)

mpl.get_backend()

plt.plot?
plt.plot(3, 2)
plt.plot(3, 2, ".")

#plt.show()
plt.figure()
plt.plot(3,2,"o")
ax=plt.gca() #get current axis, like gcf(get current figure)
ax.axis([0,6,0,10])

plt.figure()
plt.plot(3,5,"o")
plt.plot(5,7,"o")
plt.plot(1,3,"o")
ax=plt.gca() #get current axis
ax.get_children()

# matplotlib usually has 2 layers: backend layer, artist layer
# scripting layer is to evaluate specialist: simplifies access to the artist and backend layers.

### Scatterplots
import numpy as np
x=np.array(range(1,9))
y=x
plt.figure()
plt.scatter(x,y)

colors = ["green"] * (len(x)-1)
colors.append("red")
plt.figure()
plt.scatter(x,y, s= 100, c=colors) # size =100

zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
list(zip_generator)

zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
x1, y1 =zip(*zip_generator) # unzip, with 5 parameters with length of 2 for each
# if you pass a list or an interval to a func and prepare it with *,
# each item is taken out of the iterable and passed as a separate argument.
print(x1)
print(y1)
plt.figure()
plt.scatter(x[:2], y[:2], s=100, c="blue", label = "Tall Students")
plt.scatter(x[2:], y[2:], s=100, c="red", label = "Short Students")
plt.xlabel("this is of times the child kicked a ball")
plt.ylabel("the grade of student")
plt.title("title example")
plt.legend()
plt.legend(loc=4, frameon=False, title="legend")
my_legend = plt.gca().get_children()[-2]
my_legend.get_children()[0].get_children()[1].get_children()[0].get_children()

from matplotlib.artist import Artist
def rec_gc(art, depth=0):
    if isinstance(art, Artist):
        print("    " * depth + str(art))
        for child in art.get_children():
            rec_gc(child, depth+2)
        pass #next child
    pass #end if
rec_gc(my_legend)

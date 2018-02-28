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



##Line Plots
linear_data = np.array(range(1,9))
quadratic_data = linear_data**2
plt.figure()
plt.plot(linear_data,"-o", quadratic_data, "-o")
plt.plot([22,44,55], "--r")
plt.legend(list("abc"))

plt.gca().fill_between(range(len(linear_data))
                       ,linear_data,quadratic_data,facecolor="blue"
                       , alpha = 0.25)


plt.figure()
date_list = np.arange("2017-01-01", "2017-01-09", dtype="datetime64[D]")
plt.plot(date_list, linear_data, "-o", date_list,quadratic_data, "-o")

plt.figure()
date_list = np.arange("2017-01-01", "2017-01-09", dtype="datetime64[D]")
date_list = list( map(pd.to_datetime, date_list) )
plt.plot(date_list, linear_data, "-o", date_list,quadratic_data, "-o")

x=plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(45)

plt.subplots_adjust(bottom=0.25)

ax=plt.gca()
ax.set_xlabel("Date")
ax.set_ylabel("Units")
ax.set_title("Quadratic vs.Linear Performance")

# latex format title
ax.set_title("Quadratic ($x^2$) vs. Linear ($x$) Performance")



### Bar Charts
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3)

new_xvals = []
for item in xvals:
    new_xvals.append(item+0.3)

plt.bar(new_xvals, quadratic_data, width =0.3, color="red")

from random import  randint
linear_err = [randint(0,15) for x in range(len(linear_data))]
plt.bar(xvals, linear_data, width=0.3, yerr=linear_err)

#stack bar chart
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data,width =0.3, color ="b")
plt.bar(xvals, quadratic_data, width=0.3, bottom = linear_data, color="r")

#pivot chart to horizontal
plt.figure()
xvals = range(len(linear_data))
plt.barh(xvals, linear_data,height =0.3, color ="b")
plt.barh(xvals, quadratic_data, height=0.3, left=linear_data, color="r")


#Dejunkifying a plot
plt.figure()
languages = ["Python", "SQL", "Java", "C++", "JavaScript"]
popular = [56, 39, 34, 34, 29]
xvals=range(len(popular))
#plt.bar(languages, popular) #sorted by languages
plt.bar(xvals, popular, width=0.3, color="b", align ="center")

plt.xticks(xvals, languages, alpha=0.8)
#remove ylabel
#plt.ylabel('% Popularity')
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

#TODO: remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# TODO: remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# change the bar colors to be less bright blue
bars = plt.bar(xvals, popular, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, str(int(bar.get_height())) + '%',
                   ha='center', color='w', fontsize=11)
plt.show()
import game_utils as g
import random as r

start, end = 1, 100
computer = r.randint(start,end)
no = True
count = 0
while no :
    human = g.get_data(start, end)
    count += 1
    no = g.is_right(human, computer,count)
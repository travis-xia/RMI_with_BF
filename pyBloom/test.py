from pyBloom.Newstructure import Filter

Tree=Filter(0.001)
for i in range(10):
    Tree.Add(i)
Tree.Add(1<<17)
for i in range(20):
    print(Tree.Contains(i))

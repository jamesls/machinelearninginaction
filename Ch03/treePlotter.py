"""
Created on Oct 14, 2010

@author: Peter Harrington
"""
import matplotlib.pyplot as plot


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += get_num_leafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs


def get_tree_depth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + get_tree_depth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plot_node(nodeTxt, centerPt, parentPt, nodeType):
    create_plot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plot_mid_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString)


def plot_tree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = get_num_leafs(myTree)  #this determines the x width of this tree
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs))/2.0/plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plot_tree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(tree):
    figure = plot.figure(1, facecolor='white')
    figure.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plot.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0;
    plot_tree(tree, (0.5,1.0), '')
    plot.show()


def retrieve_tree(i):
    listOfTrees =[
        {'no surfacing':
         {0: 'no',
          1: {'flippers':
           {0: 'no',
            1: 'yes'}}}},
        {'no surfacing':
         {0: 'no',
          1: {'flippers':
              {0:
               {'head':
                {0: 'no',
                 1: 'yes'}},
               1: 'no'}}}}
    ]
    return listOfTrees[i]


def main():
    tree = retrieve_tree(1)
    create_plot(tree)


if __name__ == '__main__':
    main()

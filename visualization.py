import matplotlib.collections

import matplotlib.pyplot as plt

class Plot:
    def __init__(self, network, input_sample):
        self.nn = network
        self.circles = []
        self.lines = []
        self.sizes = self.getInfo(input_sample)
        self.width = (len(self.sizes))*4
        self.height = max(self.sizes[1:])
        self.big_inputs = False
        if self.sizes[0] > self.height:
            self.sizes[0] = self.height
            self.big_inputs = True
        self.genCircles()
        self.getLines()

    def getInfo(self, row):
        layer_sizes = [len(row)]
        for layer in self.nn.layers:
            layer_sizes.append(len(layer.neurons))
        return layer_sizes

    def genCircles(self):
        self.circles = []
        for l in range(len(self.sizes)):
            self.circles.append([])
            for i in range(self.sizes[l]):
                x, y = (l+1)*4, (i+1) + self.height/2 - self.sizes[l]/2
                self.circles[l].append((x, y))

    def getLines(self):
        self.lines = [] # [ [ (x1,x2), (y1, y2) ], [], [] ... ]
        for l in range(len(self.circles)):
            if l < len(self.circles)-1:
                for a in self.circles[l]:
                    for b in self.circles[l+1]:
                        self.lines.append([ (a[0], b[0]), (a[1], b[1]) ])

    def plotNetwork(self, current_image):
        fig, ax = plt.subplots()
        patches = []
        for layer in self.circles:
            if layer == self.circles[0]:
                if self.big_inputs:
                    for i in range(len(layer)):
                        if i < len(layer)/2:
                            patches.append( plt.Circle(layer[i], radius=0.4, ec="black", fc='white', linewidth=1) )
                        elif i > len(layer)/2:
                            patches.append( plt.Circle(layer[i], radius=0.4, ec="black", fc='white', linewidth=1) )
                        else:
                            ax.text(layer[i][0]-1.5, layer[i][1], "...", fontsize=20, fontweight="bold")
                else:
                    for i in range(len(layer)):
                        patches.append( plt.Circle(layer[i], radius=0.4, ec="black", fc='white', linewidth=1) )
            else:
                if layer == self.circles[-1]:
                    for i in range(len(layer)):
                        output = self.nn.layers[self.circles.index(layer)-1].neurons[i].output
                        patches.append(plt.Circle(layer[i], radius=0.45, ec="black", fc=(0,0,output), linewidth=1))
                        ax.text(layer[i][0]+0.5, layer[i][1]-0.3, f"{i}", fontsize=16)
                else:
                    for i in range(len(layer)):
                        output = self.nn.layers[self.circles.index(layer)-1].neurons[i].output
                        patches.append(plt.Circle(layer[i], radius=0.45, ec="black", fc=(0,output,0), linewidth=1))

        ax.set_aspect('equal')
        ax.set_ylim(top=self.height+1)
        ax.set_xlim(right=self.width+1)
        ax.axis('off')

        coll = matplotlib.collections.PatchCollection(patches, match_original=True)
        ax.add_collection(coll)

        for line in self.lines:
            ax.plot(line[0], line[1], color=(0.5,0.5,0.5), zorder=0, linewidth=0.5)

        plt.figure(200)
        plt.imshow(current_image.reshape(28,28), cmap='gray')
        plt.axis('off')

        plt.show(block=False)
        plt.pause(3)
        plt.close('all')


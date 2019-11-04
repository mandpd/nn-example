# nn-example

This is a slightly modified version of Andrej Kaparthy's convnetjs 2d example. You can see the original by clicking on the 'Source' link on the example page, or clicking [here](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html).

## Running the Example

Open the nn_example.html file in a web browser

## What the page shows

The original example has been recast to illustrate how a neural network might model the structure in a set of images. The goal is to be able to distinguish the cat images in the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image set from the dog images.

### The left hand graph

In this example we are only using the first two grayscale pixel values from each image to train the neural net. The left hand graph shows how the pixel values map onto a feature space with the first pixel value mapped on the vertical axis and the second pixel value mapped on the horizontal axis. Pixel values of zero map to the top-left corner of the graph.

You can use the data buttons below the graph to simulate different types of structure in the feature space. When first loaded, the feature space reflects a clear difference in the values of the pixels of dog images (colored green) compared with cat images (colored red) - the dog pixels cluster around the middle values of the pixel grayscales, while the cat pixels have larger or smaller values on the pixel grayscales. Visually this shows up as an inner circle of dog images surrounded by an outer circle of cat images.

### The right hand graph

If the neural net can discover this structure, then it can use it to predict the classification of images (dog vs cat) that it has not yet seen, assuming that the pattern in this training data is a reflection of the same structure in the image population as a whole.

The right hand graph provides a window into the layers that make up the neural net. The buttons below the graph allow you to switch the window between the layers that make up the neural net.

Each layer of the neural net is made up of a collection of neurons that receive input from the previous layer, modify the input values by linear (affine) transformations of the input space, and apply an activation function to the values that introduce non-linear transformations to the input space. The resulting output space of values is then forwarded to the next layer for processing.

The 'cycle' button cycles through the neurons in the selected layer showing how the feature space is processed by each neuron.


### The reset button

The neural net can be reset by clicking the reset button in the player controls section. Random values are chosen for the starting position of the net. You will notice that the resulting rotations and skews to the feature space change each time you reset the net.

### Running the model for First Time 

First time through it can be most instructive to select the final layer to view. This is annotated (fc2). This final layer includes a logistic classifier, (the default is 'softmax'), which converts the final output space into a probability that the net has assigned to each original datapoint as to whether it is from a dog image or a cat image. The probability of the datapoint being from a dog image is shown on the vertical axis. The probability runs from 0 (definitely not a dog image), to 1 (definitely a dog image). Similarly the horizontal axis shows the cat image probability between 0 and 1.

A perfect classification of the training data would have all the dog datapoints (colored green) at one corner and all the cat datapoints at the other corner. However, we need the net to also classify data that it has not seen. If the net is too tightly bound to the training data, any variations in new data might cause it to mis-classify the data. For this reason the optimal distribution will achieve a clear boundary between the different classes of datapoints that, if we could reverse the prediction space back into the original feature space, would map onto the major structure(s) in the original feature space. One of the benefits of the convnetjs library is that we can indeed do this as will be shown later.

### The play/pause button

The model can be started and paused by clicking the play/pause button in the player controls section. Once the model is started it will begin to analyze the input datapoints shown in the left hand graph. It will run the datapoints through the neural network layers creating a modified prediction. This modified prediction is evaluated for its quality using a loss function (the default is 'SGDTrainer') which creates feedback information that propagates back through the layers, modifying the transformation and threshold values in each layer in order to improve the predictions. The datapoints are then run through the modified neural net, and the process repeated. The value of the loss function is shown at the top of the graph and is updated on each pass, known as an epoch, through the net.

Once the model is running you will see the shape of the prediction space changes as the loss function evaluates and the weights and thresholds are modified. You will notice after a certain time the space is not changing, or is changing very slightly. At this point you should see the red datapoints clustered together close or at one corner of the graph, and the green datapoints similarly clustered closer to the other corner of the graph.

You can now press the play/pause button to stop the neural net iterating further. 

### Visualize the classification boundary on the original feature space

By clicking on the toggle button that shows 'off' you can enable a visualization of how the prediction space maps back onto the original feature space. This illustrates how this version of the neural net would classify new datapoints based on their first two pixel values.

### Using the model to classify unseen images

When a new data point's pixel values are plotted on this left hand graph, if the point is within the red (green) prediction space the model would predict that the data point is from a cat (dog) image.

If the structure that the neural net has found is indeed present in the image population as a whole, then we can expect it to be a good classifier of unseen cat and dog images.

### Discovering more about how the neural net works

Try the following to get further insights into how the neural net behaves:

1. Using the circle data with the prediction colors toggled 'On' and the model paused, try resetting the model multiple times.
Each time the model is reset the neural net begins with a new set of random values. The prediction colors show the initial prediction of the model. If you now play the model you can see how for each epoch of the network the prediction space changes. You'll see how the los function drives the prediction boundary around clusters of similar classed data points.

2. Do the same with the spiral data.
You should find that on some runs of the model it is unable to identify all parts of the spiral structure. THis is because the neural net is not able to model the more complex structure with its existing layers. The text area at the bottom left describes the composition of the neural network model that's being used. You can try pausing the model, modifying this description, resetting the model, and play the model again. The model will now evolve using the new neural network that you have described.
For example, you can increase the number of neurons in the first and second hidden layers by changing the current `num_neurons` value of 4 to 16. If you run this model, you should find that the additional complexity in the hidden layers is sufficient to capture the more complex structure in the spiral data.

3. Try further modifications to see their effect on how the neural net evolves. You can try modifying the activation function (default is 'relu'), classifier (the default is 'softmax'), and optimizer (the default is 'SGDtrainer'), along with all the paramters associated with the layers and optimizer.

4. Explore the rest of the examples of the convnetjs library. You can even see how a convolutional neural network (which incorporates the types of layers in this model along with additional layer types that help to improve the image classification) does against the CIFAR-10 dataset itself, [here](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) (Warning - this can overwhelm lower powered systems).


For more information on how this description is composed, please consult the original convnetjs [documentation](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html).
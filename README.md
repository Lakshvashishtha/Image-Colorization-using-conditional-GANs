# Image-Colorization-using-conditional-GANs
The task of colourizing black and white photographs necessitates a lot of human input and hardcoding. The goal is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black and white image as input and producing a colourized image as output.
Prerequisites-
(i) Know the difference RGB and Lab

(ii) Knowledge of basic Python and numpy

(iii) Idea of GAN architecture, generator and discriminator and loss functions 

##How we going to solve the problem

Image-to-Image Translation with Conditional Adversarial Networks paper, which we know by the name pix2pix, proposed a general solution to many image-to-image tasks in deep learning which one of those was colorization. In this approach two losses are used: L1 loss, which makes it a regression task, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner (by assigning the outputs a number indicating how "real" they look!). As mentioned earlier, we are going to build a GAN (a conditional GAN to be specific) and use an extra loss function, L1 loss. Let's start with the GAN.
As we know, in a GAN we have a generator and a discriminator model which learn to solve a problem together. In our setting, the generator model takes a grayscale image (1-channel image) and produces a 2-channel image, a channel for *a and another for *b. The discriminator, takes these two produced channels and concatenates them with the input grayscale image and decides whether this new 3-channel image is fake or real. Of course the discriminator also needs to see some real images (3-channel images again in Lab color space) that are not produced by the generator and should learn that they are real.
So what about the "condition" we mentioned? Well, that grayscale image which both the generator and discriminator see is the condition that we provide to both models in our GAN and expect that the they take this condition into consideration.
Let's take a look at the math. Consider x as the grayscale image, z as the input noise for the generator, and y as the 2-channel output we want from the generator (it can also represent the 2 color channels of a real image). Also, G is the generator model and D is the discriminator. Then the loss for our conditional GAN will be: 
![image](https://user-images.githubusercontent.com/90766665/176645012-a8c1c8ec-e6f3-4a1f-936f-abdd77a36161.png)

Notice that x is given to both models which is the condition we introduce two both players of this game. Actually, we are not going to feed a "n" dimensional vector of random noise to the generator as you might expect but the noise is introduced in the form of dropout layers in the generator architecture.
#Loss function we optimize
The earlier loss function helps to produce good-looking colorful images that seem real, but to further help the models and introduce some supervision in our task, we combine this loss function with L1 Loss (you might know L1 loss as mean absolute error) of the predicted colors compared with the actual colors:
![image](https://user-images.githubusercontent.com/90766665/176645199-79d2a2b7-a932-41af-8c98-5946f6d45d7d.png)

If we use L1 loss alone, the model still learns to colorize the images but it will be conservative and most of the time uses colors like "gray" or "brown" because when it doubts which color is the best, it takes the average and uses these colors to reduce the L1 loss as much as possible (it is similar to the blurring effect of L1 or L2 loss in super resolution task). Also, the L1 Loss is preferred over L2 loss (or mean squared error) because it reduces that effect of producing gray-ish images. So, our combined loss function will be: 
![image](https://user-images.githubusercontent.com/90766665/176645225-b4f85c03-0768-4942-8b38-0f5b6bd097c6.png)
where λ is a coefficient to balance the contribution of the two losses to the final loss (of course the discriminator loss does not involve the L1 loss).

###Steps in coding required for project

1. CODING THE BASELINE

(i)  Load image path

(ii) Preparing colllab to run the code

(iii) Making datasets and dataloaders

(iv) Generator proposed by pix2pix

(v) Discriminator

(vi) GAN Loss

(vii) Model Initialization

(viii)  Main model( bring together all previous parts)

(ix) Training function

 its output is far from something appealing and it cannot decide on the color of rare objects.So, it seems like that with this small dataset we cannot get good results with this strategy. Therefore, we change our strategy!
 
2.  Final model

To overcome the last mentioned problem, we pretrain the generator separately in a supervised and deterministic manner to avoid the problem of "the blind leading the blind" in the GAN game where neither generator nor discriminator knows anything about the task at the beginning of training.
Actually I use pretraining in two stages: 1- The backbone of the generator (the down sampling path) is a pretrained model for classification (on ImageNet) 2- The whole generator will be pretrained on the task of colorization with L1 loss. In fact, I'm going to use a pretrained ResNet18 as the backbone of my U-Net and to accomplish the second stage of pretraining, we are going to train the U-Net on our training set with only L1 Loss. Then we will move to the combined adversarial and L1 loss, as we did in the previous section.

(i) Using new generator

(ii) Pretaining the generator for colorization

(iii) Putting everything together again

Now we can generate good results.
The source of the noise in the architecture of the generator proposed by authors of the paper pix2pix was the dropout layers. However, when I investigated the U-Net we built with the help of fastai, I did not find any dropout layers in there! Actually I first trained the final model and got the results and then I investigated the generator and found this out.
This conditional GAN can still work without dropout but the outputs will be more deterministic because of the lack of that noise; however, there is still enough information in that input grayscale image which enables the generator to produce compelling outputs.

I have uploaded the colab file along with this file. I have trained the outputs and have written everything systematically in the notebook.
Colab file is open with private outputs because when I first trained the model it was fine but the second time when I tried it from start it was giving some errors , so I decided to set it with private outputs. 

I have also uploaded stepwise codes in case file is too large

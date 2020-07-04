


# Table of Contents

1.  [*Problem*](#org9124e2a)
2.  [*High Level Steps*](#org2b3434d)
    1.  [*Recommended Overview  of the steps generally followed*](#org25df29c)
        1.  [**This problem consists of two sub problems:**](#org8d922c8)
3.  [Scoring](#org78f300b)
4.  [*Content Image* : Japanese-garden](#org7199751)
5.  [*Style image*   :  Picasso-style](#orgbdd1c6b)



<a id="org9124e2a"></a>


# CHALLENGE EXERCISE
###  *Problem*

Objective  is to apply the style of an image, which we will term as
"style image" to a target image while preserving the content of the
target image.

-   ***Style*** is textures and visual patterns in an image. Example is brush strokes of an artist
-   ***Content*** is the macro structure of an image, like buildings, people, objects in the content of the image.



<a id="org2b3434d"></a>

###  *High Level Steps*

-   choose the image to style
-   choose the style reference image (here we have provided a Picasso style image)
-   choose a pre-trained deep neural network (CNN type) and obtain feature representations of intermediate layers. This step is done
    to achieve the representations of both the content image and style image. For the content image, best option is to obtain the feature
    representations of highest layers as they would contain information on the image macro-structure. For the style reference image, feature
    representations are obtained from multiple layers at different scales.
-   define loss function
-   **Loss function** should be taking into account the **Content-loss**, **Style-loss**, **Variation-loss**.
-   _*Optimize on each iteration to minimize the loss*_



<a id="org25df29c"></a>

## ___**Recommended Overview  of the steps generally followed**___

-   Create a random input image
-   Pass the input through a pre-trained backbone architecture 
-   Calculate loss and compute the gradients w.r.t input image pixels. Hence only the input pixels are adjusted whereas the
    weights remain constant.


<a id="org8d922c8"></a>

##  ___**This problem consists of two sub problems:**___

1.  *to generate the content*    and 

2.  *to generate the style*

###  *SubProblem — 1. Generate Content* 

    The problem is to produce an image that contains a content as in the
    content image.
    
    **General Guideline** 
    One point to note here is that the image should only contain the
    content(as in a rough sketch of the content image and not the texture
    from the content image, since the output should contain a style
    different from the style image)

###  *SubProblem — 2. Generate Style*

    The problem is to produce an image which contains the style as in the
    style image.  
    
    **General Guideline**
    Compute ****MSE loss between gram matrix of input and the style image**** and
    you should be good to generate an input image with the required style.
    Publish the model details, output results and model metrics.
    
###     *Requirement*
    
    1.  We require that  **Python3.x** is the coding language to be used for model development.
    2.  The code submission has to be as a self contained **Python Jupyter Notebook.**
    3.  Provide  URLs of imported  modules that are not part of  Python Standard Library.
    4.  Your code should be documented.
    5.  Label any figures if included.
    6.  Provide proper citations to any referenced authors' material with credits.
    7.  Include a summary of the approach to the exercise, your learnings and any takeaways.


<a id="org78f300b"></a>

###    *Scoring*

Each participating team will be scored on the implementation of the machine learning model. 
The short listed teams will undergo tcon to guage their understanding of the nuances of the model and its architecture, 
understanding of library calls invoked, and expertise of language constructs.

###    *SUBMISSION OF SOLUTION*

1. How to get started : 
   It is assumed if you have reached this page, you know how to connect to GitHub and clone this repository.
   If not, follow these basic steps on your localhost :
```bash
$ mkdir my_challenge_project
$ cd my_challenge_project
$ git clone git@github.com:myelin-foundry/challenge.git  # if you are using SSH, or 
$ git clone https://github.com/myelin-foundry/challenge.git  # if you are using HTTPS
```

  This should create the subdirectory 'challenge' under  the directory 'my_challenge_project'  on your localhost with all 
  the necessary files.

2. Now that you have an exact clone of the instructions and the image files, you work on your localhost to create the model 
   which will address what this challenge exercise is about. As mentioned above, you need to use Python 3.x as your programming 
   language and develop the solution using Jupyter Notebook format provided under Python Standard Library.  

3. Once your problem is solved, you can submit your solution ( before the closing date which was provided in the instruction
   document in the welcome email) by either 
   - emailing the  jupyter notebook to vasant@myelinfoundry.com, or 
   - creating your personal GitHub repo, uploading your solution as jupyter notebook to your  created repo and email the
     GitHub repo link to vasant@myelinfoundry.com

<a id="org7199751"></a>
# *Content Image* : [Japanese_garden](https://github.com/myelinfoundry-2019/challenge/blob/master/japanese_garden.jpg)
<a id="orgbdd1c6b"></a>
# *Style image*   :  [Picasso_selfportrait](https://github.com/myelinfoundry-2019/challenge/blob/master/picasso_selfportrait.jpg)


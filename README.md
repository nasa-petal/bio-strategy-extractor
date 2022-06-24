# bio-strategy-extractor

## What Does This Project Do?

The goal is to research a method that will allow us to conduct text extraction and summarization of biological strategies contained in research papers or other data sources.  

## Why Is This Project Useful?

We want to be able to build a database of biological strategies grouped by function. We will be using AskNature's curated list of biological strategies [^1] when training a machine learning model. This has yet to be accommplished.

## Preliminary Results

### What Are The Inputs & Outputs?

1. We want to first extract key phrases before summarizing. We do this using Rake (a natural language toolkit). The sample input text for text extraction/summarization tool: 

"While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements, and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications”. 

2. The output result for text extraction:

"['specialized undulated surface structure',
 'structure effectively changes',
 'potential biomimetic applications',
 'pinniped sensory biology',
 'meters per second',
 'harbor seals possess',
 'using force measurements',
 'vortex street behind',
 'induced vibrations ).',
 'harbor seal whiskers']"

3. The second output result for text summarization:

"Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure." 

I combined both the output results in the last line of code so we can do a side-by-side comparison to understand the functionalities of the two different methods. We want to include text summarization, so we perform an extraction-based approach where we search the document for key sentences and phrases. 

[UPDATE] The above are old results. These are the new results for text summarization including key features of RAKE:
"Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown"

We can conclude that this uses a lot more phrases in its text. The next step is to automate these results and have a neural network train on it.

### Were The Outputs Expected?

The outputs are expected, but more work can be done to combine the two pieces of code so that the text extraction and summarization of biological strategies accurately do what is expected. This is the expected sentence for this example: 

" "A small diameter fiber with an undulated surface structure reduces vibrations caused by drag forces" which belongs to the functions "Move, Active Movement, Actively move through liquid" and "Maintain structural integrity, Manage Structural Forces, Manage Drag/Turbulence" ".

[UPDATE] The results were what was to be expected. The main goal now is to incorporate this piece into a neural network.

### What Improvements Could Be Made?

We have yet to test all of AskNature's curated list of biological strategies. This is a work-in-progress and we will be making it more automated for future use.

## How Can Users Get Started On This Project?

- First and foremost, make sure you have installed Anaconda [^2] onto your local laptop.
- Next, you can create a directory via your terminal.
- Then, navigate to your directory and git clone this repository into it. 
- Type in your terminal "jupyter notebook". This should open up the web browser showing your home directory. 
- Go to the directory you git cloned this repository in. 
- Click on "Rake_nltk_extracttxt.ipynb". You can run all lines of code and insert the text where it says "txt" (this is found in the first code block). The first code block shows what packages that need to be installed when running the entire notebook. Please install the packages first, before running the notebook.

Feel free to try optimizing this code and test AskNature's curated list of biological startegies. This code is still a work in progress. 

If there are any reccomended changes you would like to make, please create an "Issue" on GitHub (For more information please refer to "Where Can Users Get Help" section of this README). 

## Where Can Users Get Help?

You can create a "New Issue" in the issues section of GitHub. Please refer to the pictures below for the steps on how to create an issue.

1. On the top bar on GitHub click on "Issues":
<img width="440" alt="Screen Shot 2022-06-13 at 10 09 56 AM" src="https://user-images.githubusercontent.com/85698928/173408163-7ccca11d-c93e-41f8-a6be-205712bfc50e.png">
2. You should be taken to the page below. Click on "New Issue".
<img width="1440" alt="Screen Shot 2022-06-13 at 10 10 30 AM" src="https://user-images.githubusercontent.com/85698928/173408295-cee431d2-3048-453f-9a57-97e300f6421b.png">
3. Lastly, type a title letting me know what the issue is pertaining to. Note: For any new changes you would request to implement, please type "Changes to code" in the title. In the description let us know what the change is and how we can implement it. Any other details you can give would be helpful.
<img width="1440" alt="Screen Shot 2022-06-13 at 10 10 38 AM" src="https://user-images.githubusercontent.com/85698928/173408865-5baebce7-5c3f-4b12-944d-ab004bb8f143.png">

## Sources Used For This Project

- https://huggingface.co/spaces/ml6team/keyphrase-extraction
- https://toolbox.biomimicry.org/methods/abstract/
- https://toolbox.biomimicry.org/methods/discover/
- https://toolbox.biomimicry.org/methods/abstract/
- https://towardsdatascience.com/simple-text-summarization-in-python-bdf58bfee77f
- https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f
- https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

[^1]: This is AskNature's curated list of biological strategies: https://asknature.org/biological-strategies/
[^2]: You should be able to download Anaconda for free here: https://www.anaconda.com/products/distribution


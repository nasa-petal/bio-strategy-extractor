# bio-strategy-extractor

## What Does This Project Do?

The goal is to research a method that will allow us to conduct text extraction and summarization of biological strategies contained in research papers or other data sources.  

## Why Is This Project Useful?

We want to be able to build a database of biological strategies grouped by function. We will be using AskNature's curated list of biological strategies [^1] when training a machine learning model. We also made use of FOBIE and golden.json in our models. golden.json is a curated list of petalai.org biomimcry papers.

## Preliminary Results for RAKE

### What Are The Inputs & Outputs?

1. We want to first extract key phrases before summarizing. We do this using Rake (a natural language toolkit). For more information on the code, you can view the sources section. The code was used and combined from the sources. The sample input text for text extraction/summarization tool: 

"While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements, and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications”. 

2. The output result for text extraction (for RAKE):

"['specialized undulated surface structure',
 'structure effectively changes',
 'potential biomimetic applications',
 'pinniped sensory biology',
 'meters per second',
 'harbor seals possess',
 'using force measurements',
 'vortex street behind',
 'induced vibrations ).',
 'harbor seal whiskers'...]"

3. The second output result for text summarization (RAKE):

"Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure." 

I combined both the output results in the last line of code so we can do a side-by-side comparison to understand the functionalities of the two different methods. We want to include text summarization, so we perform an extraction-based approach where we search the document for key sentences and phrases. 

[UPDATE] The above are old results. These are the new results for text summarization including key features of RAKE:
"Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown"

We can conclude that this uses a lot more phrases in its text. 

### Were The Outputs Expected for RAKE?

The outputs are expected, but more work can be done to combine the two pieces of code so that the text extraction and summarization of biological strategies accurately do what is expected. This is the expected sentence for this example: 

" "A small diameter fiber with an undulated surface structure reduces vibrations caused by drag forces" which belongs to the functions "Move, Active Movement, Actively move through liquid" and "Maintain structural integrity, Manage Structural Forces, Manage Drag/Turbulence" ".

[UPDATE] The results were what was to be expected. The main goal now is to incorporate this piece into a neural network.

### What Improvements Could Be Made for RAKE?

We have yet to test all of AskNature's curated list of biological strategies. We did test this on a couple petalai.org biomimicry papers and RAKE was able to output keywords/phrases. I used RAKE components to come up with the following resulting output (this is for one of the petalai abstract papers): "The 4 fibrous proteins of honeybee silk are small (∼30 kDa each) and nonrepetitive and adopt a coiled coil structure. Each species produced orthologues of the 4 small fibroin proteins identified in honeybee silk
There was extensive sequence divergence among the bee and ant silk genes (<50% similarity between the alignable regions of bee and ant sequences), consistent with constant and equivalent divergence since the bee/ant split (estimated to be 155 Myr). None"

The abstract used from petalai.org was: "Silks are strong protein fibers produced by a broad array of spiders and insects. The vast majority of known silks are large, repetitive proteins assembled into extended β-sheet structures. Honeybees, however, have found a radically different evolutionary solution to the need for a building material. The 4 fibrous proteins of honeybee silk are small (∼30 kDa each) and nonrepetitive and adopt a coiled coil structure. We examined silks from the 3 superfamilies of the Aculeata (Hymenoptera: Apocrita) by infrared spectroscopy and found coiled coil structure in bees (Apoidea) and in ants (Vespoidea) but not in parasitic wasps of the Chrysidoidea. We subsequently identified and sequenced the silk genes of bumblebees, bulldog ants, and weaver ants and compared these with honeybee silk genes. Each species produced orthologues of the 4 small fibroin proteins identified in honeybee silk. Each fibroin contained a continuous predicted coiled coil region of around 210 residues, flanked by 23–160 residue length N- and C-termini. The cores of the coiled coils were unusually rich in alanine. There was extensive sequence divergence among the bee and ant silk genes (<50% similarity between the alignable regions of bee and ant sequences), consistent with constant and equivalent divergence since the bee/ant split (estimated to be 155 Myr). Despite a high background level of sequence diversity, we have identified conserved design elements that we propose are essential to the assembly and function of coiled coil silks."

RAKE extracted these keywords/phrases: "['23 – 160 residue length n',
 'continuous predicted coiled coil region',
 'ant silk genes (< 50',
 '4 small fibroin proteins identified',
 'small (∼ 30 kda',
 'radically different evolutionary solution',
 'identified conserved design elements',
 'ant sequences ), consistent',
 'strong protein fibers produced',
 'extensive sequence divergence among',
 'found coiled coil structure',
 '4 fibrous proteins',
 'coiled coil structure',
 'repetitive proteins assembled',
 'species produced orthologues',
 'equivalent divergence since',
 'coiled coil silks',
 'high background level',
 'around 210 residues',
 '155 myr ).',
 'honeybee silk genes',
 'silk genes',
 'subsequently identified',
 'ant split',
 'fibroin contained',
 'coiled coils',
 'honeybee silk',
 'honeybee silk',
 'sequence diversity',
 'vast majority',
 'unusually rich',
 'sheet structures',
 'parasitic wasps',
 'known silks',
 'infrared spectroscopy',
 'extended β',
 'examined silks',
 'building material',
 'broad array',
 'alignable regions',
 '3 superfamilies',
 'weaver ants',
 'bulldog ants',
 'found',
 'silks',
 'ants',
 'vespoidea',
 'termini',
 'spiders',
 'similarity',
 'sequenced',
 'propose',
 'nonrepetitive',
 'need',
 'large',
 'insects',
 'hymenoptera',
 'however',
 'honeybees',
 'function',
 'flanked',
 'estimated',
 'essential',
 'despite',
 'cores',
 'constant',
 'compared',
 'chrysidoidea',
 'c',
 'bumblebees',
 'bees',
 'bee',
 'bee',
 'bee',
 'assembly',
 'apoidea',
 'apocrita',
 'alanine',
 'adopt',
 'aculeata']"

### Inputs and Outputs for aspire:

We now want to be able to extract key functions out of this biomimcry papers. Aspire by the allenai [^3] was used to come up with a similarity model on matching fine-grained aspects of text. I have used their example demo to see if any of the abstracts (the sample Harbor Seals abstract and a couple abstracts from the golden.json file work and these are the results. The code starts off by importing the required packages, preparing the data/example abstracts, embedding it, and visualizing the optimal transport plans for the computed sentence vectors. The resulting plots are optimal transport plans for the example pairs of abstracts.

The algorithm learns fine-grained document similarity models using co-citations in the same research paper and sentence. Then the "single-match models are learned from implicit supervision in co-citation contexts" (Mysore, Cohan, Hope 2022). Finally, "multi-match models are learned by aligning aspect representations by solving an Optimal Transport problem" (Mysore, Cohan, Hope 2022). Optimal Transport is a method for geometric computation to occur on uncertain data. The final step here is what we are observing in the plots below. We have a two candidates, which are two abstracts with titles and the Query is a way to see what the matched aspects of each of the abstract is. We see that this "method uses multiple matches with an Optimal Transport mechanism that computes Earth Mover's Distance" (Mysore, Cohan, Hope 2022. This method will help with finidng better methods of text representation. 

This first one compares the Harbor Seal petalai abstract with the first golden.json abstract. The ones after compare papers within the golden.json file.
<img width="922" alt="Screen Shot 2022-07-21 at 8 49 11 PM" src="https://user-images.githubusercontent.com/85698928/180358788-0d7aa0be-f11d-4c3d-a6a8-1a2b744b2ec7.png">
<img width="922" alt="Screen Shot 2022-07-21 at 8 49 23 PM" src="https://user-images.githubusercontent.com/85698928/180358793-e41c429f-d7fc-4af2-882e-68074333e19c.png">
<img width="922" alt="Screen Shot 2022-07-21 at 8 49 34 PM" src="https://user-images.githubusercontent.com/85698928/180358797-7c508e4c-8595-4b4d-8f98-f5ca4d58c7d8.png">
<img width="922" alt="Screen Shot 2022-07-21 at 8 49 50 PM" src="https://user-images.githubusercontent.com/85698928/180358888-e4b858c8-1588-4a42-a859-e6328101632f.png">

### Were The Outputs Expected? What Improvements Can Be Made for aspire?

The outputs were expected in comparision to the demo. Now, we should figure out a way to identify the functions and incorporate it into a summary with the functions.

### Notes on SpaCy:

SpaCy serves as a backbone to NLP algorithms. We just wanted to test out how each of its features can be used for a NLP Pipeline. The pipeline should include the following:
- Sentence segmentation: breaks the given paragraph into separate sentences.
- Word tokenization: extract the words from each sentence one by one.
- 'Parts of Speech' Prediction: identifying parts of speech.
- Text Lemmatization: figure out the most basic form of each word in a sentence. "Germ" and "Germs" can have two different meanings and we should look to solve that.
- 'Stop Words' Identification: English has a lot of filter words that appear very frequently and that introduces a lot of noise.
- Dependency Parsing: uses the grammatical laws to figure out how the words relate to one another.
- Entity Analysis: go through the text and identify all of the important words or “entities” in the text.
- Pronouns Parsing: keeps track of the pronouns with respect to the context of the sentence.

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

## Sources Used For This Project (RAKE)

- https://huggingface.co/spaces/ml6team/keyphrase-extraction
- https://toolbox.biomimicry.org/methods/abstract/
- https://toolbox.biomimicry.org/methods/discover/
- https://toolbox.biomimicry.org/methods/abstract/
- https://towardsdatascience.com/simple-text-summarization-in-python-bdf58bfee77f
- https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f
- https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

[^1]: This is AskNature's curated list of biological strategies: https://asknature.org/biological-strategies/
[^2]: You should be able to download Anaconda for free here: https://www.anaconda.com/products/distribution
[^3]: aspire/allenai: https://github.com/allenai/aspire

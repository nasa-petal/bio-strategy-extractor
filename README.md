# bio-strategy-extractor

The goal is to research a method that will allow us to conduct text extraction and summarization of biological strategies contained in research papers. The sample input text we put into the text extraction/summarization tool is: "While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements, and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications”. We want to first extract key phrases before summarizing. We do this using Rake (a natural language toolkit). 

We also want to include text summarization, so we perform an extraction-based approach where we search the document for key sentences and phrases. We performed both these methods and did a side-by-side comparison to understand the functionalities of the two different methods. The outputs are expected, but more work can be done to combine the two pieces of code so that the text extraction and summarization of biological strategies accurately do what is expected. This is the expected sentence for this example: " "A small diameter fiber with an undulated surface structure reduces vibrations caused by drag forces" which belongs to the functions "Move, Active Movement, Actively move through liquid" and "Maintain structural integrity, Manage Structural Forces, Manage Drag/Turbulence" ".

Sources used: 
https://huggingface.co/spaces/ml6team/keyphrase-extraction
https://toolbox.biomimicry.org/methods/abstract/
https://asknature.org/biological-strategies/
https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715
https://towardsdatascience.com/simple-text-summarization-in-python-bdf58bfee77f
https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f

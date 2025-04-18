Mini-GPT
===
This project consists of code that will generate a bigram language model, mirroring an early version of ChatGPT. I credit Andrej Karpathy for creating the architecture of the model, as seen in [this video](https://youtu.be/kCc8FmEb1nY?si=USOan7FBimdlvX4R). 

I built off of Andrej's work by implementing a custom, naive tokenizer. Here, tokens are either words, spaces, numbers, or punctuation marks that appear in the input text. Where Andrej's model generated predictions on a character-level (i.e. In the sentence 'I love codin' what is the most likely next character? Probably 'g' in this case), this model generates predictions at the token level (i.e. In the sentence 'I love ' what is the most likely next token? Definitely 'coding' in my case). Tokenization improves the performance of the model significantly.

The initial character-level predictions were highly chaotic. Intuitively, studying what characters are most likely to appear next to each other is pretty random. How would you discern which characters the character 'e' is most likely to be surrounded by? It could be all 26 letters, really. As such, the model would generate non-sensical text; while it would offer the appearance of words by adding spaces here and there, the actual 'words' were not words at all.

The new, token-level model makes the generation less granular. Rather than predicting characters, it tries to make sense of which tokens are most likely to appear together. Studying what words appear together is a little more achievable. For example, what tokens would the token 'How' appear before? Maybe ' ' and 'did' or ' ' and 'are', making either 'How did' or 'How are'. Similar logic follows for the next token; suppose the model chose 'How did'. The next tokens would most likely be ' ' and 'you' or ' ' and 'it'. This tokenization process abstracts the predictions, making it so that the model learns to discern word meaning and sentence structure. You'll find the code for creating the model in `gpt.py`. I've included a pre-created version in this repository; you play with the model outputs using `generate.py`. 

I credit Rick Riordan and J.K. Rowling for the datasets, which consists of your choice of either the 'Percy Jackson and the Olympians' (`input.txt`) series or the 'Harry Potter' series (`input2.txt`).

Here's a sample of what the generation looks like:
```
(venv) gazirahman@starlabs-3 minigpt % python generate.py
Enter the text that you'd like to be completed: He stepped into the castle
He stepped into the castle.
Harry felt Ron looked secondhand, echoing the marble Get at once.  Harry Snape, and the Hufflepuff potion had burned her feet.  The stairs he found smoke upward, muttering students.
Harry put the picture of the car.  And they should fly, George and Hagrid, as he noticed for a mirror Sonorus of the curtains.  Harry saw the window and tentatively very pelting of the curtains them fell, his magnificent wings was sitting examining him to go, Dobby marched had never raised her, looking sizes wanting a chirruping of tight she'd something
```



To run this repo, create a virtual environment with Python 3.12 and use `pip` to install the `requirements.txt` file. In each file that you wish to run, make sure to modify the paths according to your own device, and then proceed as desired. It will suffice to run either `python gpt.py` or `python generate.py`. Alternatively, you can experiment with the corresponding Jupyter Notebooks instead.

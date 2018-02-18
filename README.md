# text-generation

This project uses keras to train a LSTM network with writings of Nietzsche, 
allowing us to produce writings like him.


## train network

By running `train.py`, you can download the learning data and train the network.

```bash
python3 train.py
```

## data preparation

`download.py` retrieves text data of Nietzsce's writings and transforms it into training data.

+ question
: retrieves sequences of 60 letters from data, and transforms it into vector using "bag of words" approach for each letter (rather bag of letters).

+ answer
: transforms the letter next to the retrieved sequence, and transforms it into one-hot vector.

## prediction

The writings are produced from sequence of 60 letters (called seed). 
The next letter is decided probabilistically by applying a parameter called temperature to the softmax output of the model.

It is defined like this. 
```python
import numpy as np

def reweight_distribution(original_distribution, temp=0.5):
    """
    higher temps result in more unexpected(creative) results
    :param original_distribution: softmax output of model
    :param temp: temperature
    :return:  new distribution (sum equals to 1)
    """
    distribution = np.log(original_distribution) / temp
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)
```

examples of output(at 60 epochs)

```text
---generating with seed: "ally that puts questions to us here? what really
is this "wi"

-----temperature 0.2-------

ally that puts questions to us here? what really
is this "will and man as the man with the stronger and something the most consideration of the most deception of the spirit as the democration of the most consideration of the present and the world and the most property that it is the most and something in the deceives and all the moral historical same that the most deceives the soul and the most promised the present man and the most distinction of the spiri
-----temperature 0.5-------

omised the present man and the most distinction of the spirit as the propect the and
psychological profound taken in a sense is to be state of our our will to more evil and also in the man as the fine fighter inwhe development, and may be "of the most childis their promis
their tasks,
and of the progrted that there is not such a high conception its owing to the faith and was complexity. the property and so far,
indeed, and them in one in the patacle of ple
-----temperature 1.0-------

ty and so far,
indeed, and them in one in the patacle of pleasistic has for idea and when the wills, and circumlose
manifestation of him
were panableness; as and day,
please within
moraatiest bad as a deinging insk. querity.
elquer know its intellectual," be desilate
manyessogicat and above mat
meanurey it
has virtue and, mengativious manifestations elsepcender of bad preseful, without conscious childis through their his "peopative and
unjust
distance of a
-----temperature 1.2-------

hildis through their his "peopative and
unjust
distance of a foder germans.and, as
brunds and
its. which the expende all their well
philosopher tides to-upon is now gives eturability religioused men
gives ?quciomis socionalists..

eecte. stall tigner whittray
by -doinging of
the
inverceptable of easlances, highly
michowation?


5

epace of pleasant is nterd by others;--whoever inoor over.

sceour, at would borung true--he virtue prevallsons speaked at the
```

## References

+ François Chollet: Deep Learning with Python

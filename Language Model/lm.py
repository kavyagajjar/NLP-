import sys
import pandas as pd
import numpy as np
import math
import random
"""
Author: Kavya Gajjar

For the hw2-my-test.txt file:
I have selected 
- few of the quotes of shakespeare, einstein, George RR martin
- Few simple english sentences short one's
- and a paragraph from audrelorde_usesofanger_1981.txt and alice_being.txt
"""


class LanguageModel():
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing =1):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        pass
    
    def generateNgram(self, n=1):
        """ Create N grams for the given tokens
        Parameters: 
            n (int): n-gram
        Output:
            Returns the n-grams
        """
        grams = {}
        self.replacingUNK() # Replace the unknown tags and update vocab
        for i in range(len(self.tokens)-(n-1)):
            key = tuple(self.tokens[i:i+n])
            grams[key] = grams.get(key, 0) + 1
        return grams
    
    def replacingUNK(self):
        """Replacing the tokens with count <=1 by <UNK>
        Update the tokens as well as the vocabulary list
        
        Output:
            None
        """
        tokens = {}
        for word in self.tokens:
            tokens[word] = tokens.get(word,0) + 1  
        
        for i in range(len(self.tokens)):
            if tokens[self.tokens[i]] == 1:
                self.tokens[i] = '<UNK>'
                
        self.vocab = {}
        for word in self.tokens:
            self.vocab[word] = self.vocab.get(word,0) +1
        return self.tokens
    
    def createNgramCount(self):
        """Calculate the Probability for the n-grams
        Use the n and (n-1) grams to calculate the probability
        
        Output:
            Return the n-gram dictionary with the probability
        """
        n_vocab = self.generateNgram(n=self.n_gram) # n-gram counts
        m_vocab = self.generateNgram(n=self.n_gram -1) #n-1 gram counts
        vocab_size = len(self.vocab)
        
        def likelihood(n_gram, n_count):
            """
            Calculate the probability
            """
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            if self.is_laplace_smoothing: # laplace smoothing
                return (n_count + 1) / (m_count + 1 * vocab_size)
            else:
                return (n_count) / (m_count)

        return { n_gram: likelihood(n_gram, count) for n_gram, count in n_vocab.items() }
    
    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
            training_file_path (str): the location of the training data to read

        Output:
            None
        """
        # Read the training file
        with open(training_file_path,'r') as f:
            self.content = f.readlines()
        
        # Pre-processing the data and  creating tokens and creating vocabulary
        self.tokens = []
        self.vocab = {}
        for i in range(len(self.content)):
            self.tokens.append(self.content[i].split())
            for word in self.content[i].split():
                self.vocab[word] = self.vocab.get(word,0) + 1
        self.tokens  = [item for sent in self.tokens for item in sent]
        
        # If unigrams
        if self.n_gram == 1:
            # Calculate num of tokens
            self.num_tokens = 0
            for sent in self.tokens:
                self.num_tokens += 1
            
            self.grams = {}
            self.replacingUNK() # replace unk
            
            for key in self.tokens:
                self.grams[key] = self.grams.get(key, 0) + 1
            
            if self.is_laplace_smoothing:
                for w in self.grams.keys():
                    self.grams[w] = (self.grams[w] +1) / (self.num_tokens + len(self.vocab))
            else:
                for w in self.grams.keys():
                    self.grams[w] = self.grams[w] / (self.num_tokens)
        else:
            # for ngrams for n>1
            self.grams = self.createNgramCount()
        
        pass

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        if self.n_gram ==1:
            #Unigram cases
            val = 1
            words = sentence.split()
            #replacing OOV words with <UNK>
            for i in range(len(words)):
                if words[i] not in self.vocab.keys():
                    words[i] = '<UNK>'
                    
            for w in words:
                val *= self.grams[w]
            
        else:
            #ngram cases
            words = sentence.split()
            #replacing OOV words with <UNK>
            for i in range(len(words)):
                if words[i] not in self.vocab.keys():
                    words[i] = '<UNK>'
            val = 1
            #create n-1 gram for unknown combinations
            m_gram = self.generateNgram(n = self.n_gram -1)

            for i in range(len(words)- (self.n_gram -1)):
                key = tuple(words[i:i+self.n_gram])
                if self.is_laplace_smoothing:
                    num_tokens = m_gram.get(tuple(words[i:i+self.n_gram -1]),  0)
                    val *= self.grams.get(key, ((0 +1) / (num_tokens + len(self.vocab))))
                else:
                    val *= self.grams.get(key, 0)
        return val
        pass
    
    def next_best_token(self, prev):
        """
        Get the next best token based on the previous value
        
        Parameters:
            prev: Prev token/s it will be () for unigram
        """
        # don't add <s> tag later in sentence i.e. it won't be a potential candidate
        without  =  ['<s>']
        # list of potential next words
        if self.n_gram > 1:
            candidates = ((ngram[-1],prob) for ngram,prob in self.grams.items() if ngram[:-1]==prev)
        else:
            #unigram case
            candidates = ((ngram,prob) for ngram,prob in self.grams.items())
            
        #remove the blacklisted words and sort the potential words based on probability
        candidates = filter(lambda candidate: candidate[0] not in without, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        
        
        # here I have given the probability distribution to all the potential next words based on thier probability and using the 
        # np.random.choice() function I am selecting a next word
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            p = []
            sump = 0
            arr = []
            for i in range(len(candidates)):
                sump += candidates[i][1]
                arr.append(candidates[i][1])
            
            for i in range(len(candidates)):
                p.append(candidates[i][1]/sump)
            index = arr.index(np.random.choice(arr, 1, p = p)[0])
            return candidates[index]

    def generate_sentence(self,i):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        sent, prob = ["<s>"] * max(1, self.n_gram-1), 1
        while sent[-1] != "</s>":
            prev = () if self.n_gram == 1 else tuple(sent[-(self.n_gram-1):])
            next_token, next_prob = self.next_best_token(prev)
            sent.append(next_token)
            # prob *= next_prob

            if len(sent) >= 100: # i have kept 100 as maximum length to avoid timeout
                sent.append("</s>")
        # Appending the Ending tags for n-grams 
        if self.n_gram >2:
            for count in range(self.n_gram -2):
                sent.append("</s>")
        generated_sent  = ' '.join(sent)
        return (generated_sent)
        pass

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        sentences = []
        for i in range(n):
            generated_sent = self.generate_sentence(i)
            sentences.append(generated_sent)
        return sentences
        pass

    def perplexity(self, test_sequence):
        """
        Measures the perplexity for the given test sequence with this trained model. 
        As described in the text, you may assume that this sequence may consist of many sentences "glued together".

        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """ 
        perp = 0
        count = 0
        length = 0
        for sent in test_sequence:
            perp += math.log(self.score(sent))
            words = sent.split()
            # for end tags
            key = []
            for l in range(self.n_gram ):
                key.append("</s>")
            length += len(words) + len(key)
            key = ' '.join([str(elem) for elem in key])
            print(key)
            perp += math.log(self.score(key))
            count += 1
            if count ==10:
                break
        return math.pow(math.exp(perp), -(1/length))

    

def main():
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]
    
    print("="*50)
    print("Model: Unigram with laplace smoothing")
    lm = LanguageModel(1, True)
    lm.train(training_path)
    sentences = lm.generate(50)
    print("50 generated Sentences")
    for sent in sentences:
        print(sent)
    
    print("\n"*2)
    print("="*20)
    print("Test set file:", testing_path1)
    with open(testing_path1,'r') as f:
        test_content = f.readlines()
    print("# of Sentences",len(test_content))
    
    probabilities = []
    for sentence in test_content:
        probabilities.append(lm.score(sentence))
    print("Average Probability", round(np.mean(np.array(probabilities), axis = 0), 6))
    print("Standard Deviation", round(np.std(np.array(probabilities), axis = 0),6))
    print("Perplexity", round(lm.perplexity(test_content),6))
    
    print("="*20)
    print("Test set file:", testing_path2)
    with open(testing_path2,'r',encoding="utf8") as f:
        test_content = f.readlines()
    print("# of Sentences",len(test_content))
    
    probabilities = []
    for sentence in test_content:
        probabilities.append(lm.score(sentence))
    print("Average Probability", round(np.mean(np.array(probabilities), axis = 0), 6))
    print("Standard Deviation", round(np.std(np.array(probabilities), axis = 0),6))
    print("Perplexity", round(lm.perplexity(test_content),6))
    
    print("\n"*3)
    print("="*50)
    print("Model: Bigram with laplace smoothing")
    lm = LanguageModel(2,True)
    lm.train(training_path)
    sentences = lm.generate(50)
    print("50 generated Sentences")
    for sent in sentences:
        print(sent)
    
    print("\n"*2)
    print("="*20)
    print("Test set file:", testing_path1)
    with open(testing_path1,'r') as f:
        test_content = f.readlines()
    print("# of Sentences",len(test_content))
    
    probabilities = []
    for sentence in test_content:
        probabilities.append(lm.score(sentence))
    print("Average Probability", round(np.mean(np.array(probabilities), axis = 0), 6))
    print("Standard Deviation", round(np.std(np.array(probabilities), axis = 0),6))
    print("Perplexity", round(lm.perplexity(test_content),6))
    
    print("="*20)
    print("Test set file:", testing_path2)
    with open(testing_path2,'r',encoding="utf8") as f:
        test_content = f.readlines()
    print("# of Sentences",len(test_content))
    
    probabilities = []
    for sentence in test_content:
        probabilities.append(lm.score(sentence))
    print("Average Probability", round(np.mean(np.array(probabilities), axis = 0), 6))
    print("Standard Deviation", round(np.std(np.array(probabilities), axis = 0),6))
    print("Perplexity", round(lm.perplexity(test_content),6))
    
    pass

    
if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw3_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)
    main()


For the evaluation of the data frames, the following were used:

### CER and WER
To calculate the CER, WER and WER(order independent) the [ocrevalUAtion tool](https://github.com/impactcentre/ocrevalUAtion) from the Impact Centre was used.

### Jaccard coefficient
The following function was used to calculate the jaccard coefficient:

```
def get_jaccard_sim(gt, ocr):
   gt = set(gt.split())  
   ocr = set(ocr.split())  
   union = len(gt.union(ocr))
   intersection = len(gt.intersection(ocr))  
   word_error_rate = intersection / union  
   return word_error_rate
``` 

### Levenshtein distance
This calculation gave some troubles. 

For the Meertens set, I used the [Jellyfish package](https://pypi.org/project/jellyfish/) from Python. 
Then I calculated the normalized Levenshtein distance with the following code:

```
def normalized_levenshtein(str1, str2):
    dist = jellyfish.levenshtein_distance(str1, str2)
    maxstr = max(len(str1),len(str2))
    output = dist / maxstr
    return output
```

Since the jellyfish packages was really slow for the Impact set, I implemented the same algorithm in c# sharp and calculated the distance with the c# function, which performed around 50 times faster. The normalization was done in Python as described above. 
The function:
```
using System;
using System.IO;

public static class StringDistance
{
    /// <summary>
    /// Compute the distance between two strings.
    /// </summary>
    public static int LevenshteinDistance(string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[,] d = new int[n + 1, m + 1];

        // Step 1
        if (n == 0)
        {
            return m;
        }

        if (m == 0)
        {
            return n;
        }

        // Step 2
        for (int i = 0; i <= n; d[i, 0] = i++)
        {
        }

        for (int j = 0; j <= m; d[0, j] = j++)
        {
        }

        // Step 3
        for (int i = 1; i <= n; i++)
        {
            //Step 4
            for (int j = 1; j <= m; j++)
            {
                // Step 5
                int cost = (t[j - 1] == s[i - 1]) ? 0 : 1;

                // Step 6
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }
        // Step 7
        return d[n, m];
    }
}
```

### Dictionary lookup
The dictionary lookup was performed with two lexicons.  
The first was a modern lexicon of the Dutch language, provided by OpenTaal. For this lookup, the ['wordlist.txt' ](https://github.com/OpenTaal/opentaal-wordlist) was used.  
The second lexicon was from the Instituut van de Nederlandse taal and is called the [INT historical wordlist](https://ivdnt.org/taalmaterialen/102-taalmaterialen/2126-tstc-int-historische-woordenlijst-j).   Quote from the INT website: "_Two lists, each consisting of approx. 500,000 historical word forms, to be used for OCR and OCR post-correction, for the period of 1550 – 1970, approximately._"   
Both list from the INT were used. 


The following step were performed to calculate the percentage of the dictionary lookup: 

* The text was stripped from punctuation marks;
* The text and word lists were set to lowercase;
* The text was lemmatized with the lemmatize option from the [Spacy Python package](https://spacy.io/usage/spacy-101);
* The text was transformed into a set;
* The set was compared with the word lists;
* The whole numbers 0 till 1000000 were marked as correct;
* The percentage was calculated by dividing the total of matched words through the total set of words from the text.






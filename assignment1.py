import gzip
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

allStuff = []
for l in readCSV("train_Interactions.csv.gz"):
    allStuff.append(l)

ratingsTrain = allStuff
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 5), skip_lines=1)
df = pd.DataFrame(allStuff)
data = Dataset.load_from_df(df, reader=reader)

trainset = data.build_full_trainset()

model = SVD(reg_all=0.275, n_factors=50, n_epochs=40)

model.fit(trainset)

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    prediction = model.predict(uid=u, iid=b, verbose=False)
            
    predictions.write(u + ',' + b + ',' + str(prediction.est) +'\n')
    
predictions.close()

### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
  bookCount[book] += 1
  totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalRead/1.36: break

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    user,book = l.strip().split(',')
    # (etc.)
    th = 0.0026

    b_primes = ratingsPerUser[user] # books that the user has read, so we need to find all the users for each book
    b_primes = [t[0] for t in b_primes]

    users_b = ratingsPerItem[book] # users that have read the current book
    users_b = [t[0] for t in users_b]

    max_sim = 0
    for b_prime in b_primes:
        if b_prime == book: continue
        users_b_prime = [t[0] for t in ratingsPerItem[b_prime]]
        sim = Jaccard(set(users_b), set(users_b_prime))
        if sim > max_sim:
            max_sim = sim
    pred = 1 if book in return1 else 0
    if pred == 1:
        predictions.write(user + ',' + book + ",1\n")
    else:
        predictions.write(user + ',' + book + ",0\n")
predictions.close()

print("done")
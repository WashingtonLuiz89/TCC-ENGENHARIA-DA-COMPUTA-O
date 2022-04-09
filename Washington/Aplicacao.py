import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.metrics import ConfusionMatrix
from nltk.stem import RSLPStemmer
from Base import BaseTreinamento
from Base import BaseTeste
from Teste import teste

#nltk.download()


stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) 
                       for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemmerTreinamento = aplicastemmer(BaseTreinamento)
frasescomstemmerTeste = aplicastemmer(BaseTeste)

def buscaPalavras(frases):
    todasPalavras = []
    for (palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras
 
palavrasTreinamento = buscaPalavras(frasescomstemmerTreinamento)
palavrasTeste = buscaPalavras(frasescomstemmerTeste)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrasTreinamento)
frequenciateste = buscafrequencia(palavrasTeste)

def buscaPalavrasUnicas(frequencia):
    palavras = nltk.FreqDist(frequencia)
    freq = palavras.keys()
    return freq

palavrasUnicasTreinamento = buscaPalavrasUnicas(palavrasTreinamento)
palavrasUnicasTeste = buscaPalavrasUnicas(palavrasTeste)

def extratorPalavrasTreinamento(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def extratorPalavrasTeste(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTeste:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

baseCompletaTreinamento = nltk.classify.apply_features(extratorPalavrasTreinamento, frasescomstemmerTreinamento)
baseCompletaTeste = nltk.classify.apply_features(extratorPalavrasTeste, frasescomstemmerTeste)

classificador = nltk.NaiveBayesClassifier.train(baseCompletaTreinamento)
print("\nEmoções encontradas na BaseCompletaTreinamento:\n")
print(classificador.labels())
print("\nTabela de Probabilidade dos Recursos mais informativos do Classificador:\n")
print(classificador.show_most_informative_features(20))
print("\nResultado do nivel de acuracidade do Teste Realizado:\n")
print(nltk.classify.accuracy(classificador, baseCompletaTeste))

erros = []
for (frase, classe) in baseCompletaTeste:
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))

from nltk.metrics import ConfusionMatrix
esperado = []
previsto = []
for (frase, classe) in baseCompletaTeste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

#esperado = 'alegria alegria alegria alegria medo medo surpresa surpresa'.split()
#previsto = 'alegria alegria medo surpresa medo medo medo surpresa'.split()

matriz = ConfusionMatrix(esperado, previsto)
print("\nResultado do Matriz do Teste Realizado:\n")
print(matriz)


testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrasTreinamento) in teste.split():
    comstem = [p for p in palavrasTreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorPalavrasTeste(testestemming)
print("\nResultado do Stemming do Teste Realizado:\n")
print(novo)

distribuicao = classificador.prob_classify(novo)
print("\nResultado do clasificador Naive Bayes do Teste Realizado:\n")
print(classificador.classify(novo))
classnovo = classificador.classify(novo)

print("\nPorcentagem do classificador de Naive Bayes do Teste Realizado:\n")
for classe in distribuicao.samples():
   print("%s: %.5f" % (classe, distribuicao.prob(classe)))

print("\nSentenças do Tokenize do Teste Realizado:\n")
tokens = word_tokenize(teste)
tags = pos_tag(tokens)

for token, tag in zip (tokens,tags):
   print(token,tag)
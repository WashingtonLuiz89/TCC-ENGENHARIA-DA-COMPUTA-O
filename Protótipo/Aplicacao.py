# Realizado as importações e abrindo dependecias. 
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.metrics import ConfusionMatrix
from Base import BaseTreinamento
from Base import BaseTeste
from Teste import teste

#nltk.download()

#criando e atribuindo uma classe de stopwords referenciando ao NLTK em português.
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

# Realizado a Implememtação da função removestopwords.
def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

# Realizado a Implememtação da função aplicastemmer.
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

# Realizado a Implememtação da função buscaPalavras.
def buscaPalavras(frases):
    todasPalavras = []
    for (palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras
 
palavrasTreinamento = buscaPalavras(frasescomstemmerTreinamento)
palavrasTeste = buscaPalavras(frasescomstemmerTeste)

# Realizado a Implememtação da função buscafrequencia.
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrasTreinamento)
frequenciateste = buscafrequencia(palavrasTeste)

# Realizado a Implememtação da função buscaPalavrasUnicas.
def buscaPalavrasUnicas(frequencia):
    palavras = nltk.FreqDist(frequencia)
    freq = palavras.keys()
    return freq

palavrasUnicasTreinamento = buscaPalavrasUnicas(palavrasTreinamento)
palavrasUnicasTeste = buscaPalavrasUnicas(palavrasTeste)

# Realizado a Implememtação da função extratorPalavrasTreinamento.
def extratorPalavrasTreinamento(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# Realizado a Implememtação da função extratorPalavrasTeste.
def extratorPalavrasTeste(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTeste:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

baseCompletaTreinamento = nltk.classify.apply_features(extratorPalavrasTreinamento, frasescomstemmerTreinamento)
baseCompletaTeste = nltk.classify.apply_features(extratorPalavrasTeste, frasescomstemmerTeste)

classificador = nltk.NaiveBayesClassifier.train(baseCompletaTreinamento)
# Realizado a Implememtação das Impressões da Emoções encontradas na BaseCompletaTreinamento.
print("\nEmoções encontradas na BaseCompletaTreinamento:\n")
print(classificador.labels())

# Realizado a Implememtação da Tabela de Probabilidade dos Recursos mais informativos do Classificador.
print("\nTabela de Probabilidade dos Recursos mais informativos do Classificador:\n")
print(classificador.show_most_informative_features(20))

# Realizado a Implememtação do Resultado do nivel de acuracidade do Teste Realizado.
print("\nResultado do nivel de acuracidade do Teste Realizado:\n")
print(nltk.classify.accuracy(classificador, baseCompletaTeste))

# Realizado a Implememtação dos tratamentos de erros.
erros = []
for (frase, classe) in baseCompletaTeste:
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))

# Realizado a Implememtação da Matriz de confusão.
esperado = []
previsto = []
for (frase, classe) in baseCompletaTeste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)


matriz = ConfusionMatrix(esperado, previsto)

# Realizado a Implememtação das Impressões do Resultado da Matriz do Teste Realizado.
print("\nResultado do Matriz do Teste Realizado:\n")
print(matriz)

# Realizado a Implememtação do testestemming.
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrasTreinamento) in teste.split():
    comstem = [p for p in palavrasTreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorPalavrasTeste(testestemming)

# Realizado a Implememtação das Impressões Resultado do Stemming do Teste Realizado
print("\nResultado do Stemming do Teste Realizado:\n")
print(novo)

distribuicao = classificador.prob_classify(novo)

# Realizado a Implememtação das Impressões do  do classificador Naive Bayes do Teste Realizado
print("\nResultado do classificador Naive Bayes do Teste Realizado:\n")
print(classificador.classify(novo))
classnovo = classificador.classify(novo)

# Realizado a Implememtação das Impressões da porcentagem do classificador de Naive Bayes do Teste Realizado.
print("\nPorcentagem do classificador de Naive Bayes do Teste Realizado:\n")
for classe in distribuicao.samples():
   print("%s: %.5f" % (classe, distribuicao.prob(classe)))

# Realizado a Implememtação das Impressões de Sentenças do Tokenize do Teste Realizado.
print("\nSentenças do Tokenize do Teste Realizado:\n")

# Realizado a Implememtação do Tokenize.
tokens = word_tokenize(teste)
tags = pos_tag(tokens)

for token, tag in zip (tokens,tags):
   print(token,tag)
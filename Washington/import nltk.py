import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.metrics import ConfusionMatrix

#nltk.download()

BaseTreinamento = [
('eu gosto disso', 'alegria'),
('este trabalho e agradável','alegria'),
('gosto de ficar no seu aconchego','alegria'),
('fiz a adesão ao curso hoje','alegria'),
('eu sou admirada por muitos','alegria'),
('adoro como você e','alegria'),
('adoro seu cabelo macio','alegria'),
('adoro a cor dos seus olhos','alegria'),
('somo tão amáveis um com o outro','alegria'),
('sinto uma grande afeição por ele','alegria'),
('quero agradar meus filhos','alegria'),
('me sinto completamente amado','alegria'),
('eu amo você','alegria'),
('Viver feliz é viver de bem com a vida','alegria'),
('Sonhar, sorrir, viver e todo dia agradecer','alegria'),
('Melhor que rir, é rir com alguém ao seu lado','alegria'),
('Bonito é estar de bem com a vida','alegria'),
('O melhor lugar do mundo é onde você está em paz','alegria'),
('A vida é linda','alegria'),
('eu sou admirada por muitos','alegria'),
('me sinto completamente amado','alegria'),
('amar e maravilhoso','alegria'),
('estou me sentindo muito animado novamente','alegria'),
('eu estou muito bem hoje','alegria'),
('que belo dia para dirigir um carro novo','alegria'),
('o dia está muito bonito','alegria'),
('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
('o amor e lindo','alegria'),
('nossa amizade e amor vai durar para sempre', 'alegria'),
('eu amo voçê','alegria'),

(' ele a feriu profundamente ',' raiva '),
(' vou despejar minha cólera em você ',' raiva '),
(' me sinto atormentado ',' raiva '),
(' não me contrarie ',' raiva '),
(' vou destruir tudo que foi construído ',' raiva '),
(' não consigo terminar este trabalho, e muito frustrante ',' raiva '),
(' me frustra a sua presença aqui ',' raiva '),
(' esta comida me parece muito ruim ',' raiva '),
(' você me destrói ',' raiva '),
(' estamos separados ',' raiva '),
(' estou odiando este vestido ',' raiva '),
(' não pude comprar meu celular hoje ',' raiva '),
(' ela e uma garota ruim ',' raiva '),
(' estivemos em um show horroroso ',' raiva '),
(' o ingresso estava muito caro ',' raiva '),
(' se eu estragar tudo vai por água a baixo ',' raiva '),
(' não possuo dinheiro algum ',' raiva '),
(' sou muito pobre ',' raiva '),
(' vai prejudicar a todos esta nova medida ',' raiva '),
(' ficou ridículo ',' raiva '),
(' este sapato esta muito apertado ',' raiva '),
(' a musica e uma ofensa aos meus ouvidos ',' raiva '),
(' não consigo terminar uma tarefa muito difícil ',' raiva '),
(' reprovei em minha graduação ',' raiva '),
(' estou muito chateado com tudo ',' raiva '),
(' eu odeio em você ',' raiva '),
(' e um desprazer conhecê-lo ',' raiva '),
(' estou desperdiçando minhas ferias ',' raiva '),
(' e muito ruim este jogo ',' raiva '),
(' vamos ter muito rancor pela frente ',' raiva '),
(' não achei que seria tão terrível ',' raiva '),

(' magicamente você me surpreendeu ',' surpresa '),
(' e imenso esse globo ',' surpresa '),
(' isso e tremendamente interessante ',' surpresa '),
(' meu bilhete for sorteado, inacreditável! ',' surpresa '),
(' um assalto a mão armada! ',' surpresa '),
(' incrível, cabe em qualquer lugar! ',' surpresa '),
(' você por aqui? ',' surpresa '),
(' não dá pra acreditar no que ela me contou ',' surpresa '),
(' os convidados já estão chegando! ',' surpresa '),
(' puxa vida! nunca nos livramos de alguem tão depressa ',' surpresa '),
(' micha carteira sumiu, eu estava com ela na mão ',' surpresa '),
(' oh! um disco voador ',' surpresa '),
(' amigos, que bela surpresa! ',' surpresa '),
(' nunca pensei que veria isso e perto ',' surpresa '),
(' nem acredito que comi tanto ',' surpresa '),
(' não acredito que veio me ver ',' surpresa '),
(' não acredito que e tão descarado ',' surpresa '),
(' me surpreende sua falta de tato ',' surpresa '),
(' o predio onde eles moravam desabou! ',' surpresa '),
(' inacreditável um bolo tão grande ',' surpresa '),
(' e serio mesmo? não dá pra acreditar ',' surpresa '),
(' como assim não vai ao nosso encontro? ',' surpresa '),
(' como assim não tem ninguem em casa? ',' surpresa '),
(' ue, mas para onde ele foi?! ',' surpresa '),
(' por essa eu não esperava ',' surpresa '),
(' nossa, olha só que mergulho ',' surpresa '),
(' minha esposa está grávida! ',' surpresa '),
(' meu dinheiro sumiu! ',' surpresa '),
(' e verdade que os dois terminaram?!? ',' surpresa '),
(' caramba, nem vi você chegar ',' surpresa '),
(' nossa, como pode alguem cozinhar tão mal? ',' surpresa '),
(' nossa que incrível ',' surpresa '),
(' a fórmula sumiu! ',' surpresa '),
(' eu nem acredito que já estou terminando o curso ',' surpresa '),
(' não acredito que esta aqui comigo novamente ',' surpresa '),
(' está escondendo algo de nós! ',' surpresa '),
(' como assim, ainda não terminou a tarefa ',' surpresa '),
(' pensei que já estivesse pronta! ',' surpresa '),
(' opa! quem apagou a luz? ',' surpresa '),
(' caramba! aonde vai tão rápido? ',' surpresa '),
(' estamos seguindo o caminho errado! ',' surpresa '),
(' quatro reais o litro da gasolina! ',' surpresa '),
(' me assustei ao vê-lo desse jeito! ',' surpresa '),
(' minha mãe está grávida, acredita nisso? ',' surpresa '),
(' parece mentira você ter crescido tanto ',' surpresa '),
(' me surpreende sua imaginação ',' surpresa '),
(' suas roupas são realmente lindas ',' surpresa '),

('que abominável esse montro!','medo'),
('vamos alarmar a todos sobre a situação','medo'),
('estou amedrontada','medo'),
('estou com muito medo da noite','medo'),
('ele esta me ameaçando a dias','medo'),
('quanta angustia','medo'),
('estou angustiada','medo'),
('angustiadamente vou sair e casa','medo'),
('isso me deixa apavorada','medo'),
('você esta me apavorando','medo'),
('estou desconfiada de você','medo'),
('não confio em você','medo'),
('ate o cachorro está apavorado','medo'),
('estou assustado com as ações do meu colega','medo'),
('agora se sente humilhado, apavorado','medo'),
('assustou a população e provocou mortes','medo'),
('estou com dificuldades para respirar e muito assustado','medo'),
('os policiais se assustaram quando o carro capotou','medo'),
('o trabalhador e assombrado pelo temor do desemprego','medo'),
('eu estou assustado', 'medo'),

('você e abominável','desgosto'),
('abomino a maneira como você age','desgosto'),
('estou adoentado','desgosto'),
('meu pai esta adoentado','desgosto'),
('estamos todos doentes','desgosto'),
('essa situação e muito amarga','desgosto'),
('disse adeus amargamente','desgosto'),
('tenho antipatia por aquela pessoa','desgosto'),
('como pode ser tão antipática!','desgosto'),
('que horrível seu asqueroso','desgosto'),
('tenho aversão agente como você','desgosto'),
('isso tudo e só chateação','desgosto'),
('estou muito chateada com suas mentiras','desgosto'),
('tão desagradável','desgosto'),
('isso me desagrada completamente','desgosto'),
('te desagrada isso','desgosto'),
('estou com enjôos terríveis','desgosto'),
('todos estão enfermos','desgosto'),
('foi uma enfermidade terrível','desgosto'),
('isso e muito grave','desgosto'),
('não seja tão grosseiro','desgosto'),
('você fez uma manobra ilegal','desgosto'),
('sua indecente, não tem vergonha?','desgosto'),
('você e malvado com as crianças','desgosto'),
('que comentário maldoso','desgosto'),
('sem escrúpulos você manipula a tudo','desgosto'),
('sinto repulsa por você','desgosto'),
('e repulsivo a maneira como olha para as pessoas','desgosto'),
('estou indisposta','desgosto'),
('a indisposição me atacou hoje','desgosto'),
('acho que vou vomitar','desgosto'),
('tem muito vomito lá','desgosto'),
('que incomodo essa dor','desgosto'),
('não me incomode nunca mais','desgosto'),
('suas bobagens estão nos incomodando','desgosto'),
('que nojo olha toda essa sujeira','desgosto'),
('como isso está sujo','desgosto'),
('tenho náuseas só de lembrar','desgosto'),
('me sinto nauseada com o cheiro desta comida','desgosto'),
('você esta obstruindo a passagem de ar','desgosto'),
('você esta terrivelmente doente','desgosto'),
('olhe que feia esta roupa','desgosto'),
('que atitude deplorável','desgosto'),
('nossa como você e feio','desgosto'),
('muito mau tudo isso','desgosto'),
('estou desgostoso com você','desgosto'),
('você cortou o meu assunto','desgosto'),
('para que tanta chateação?','desgosto'),
('esse perfume e enjoativo','desgosto'),
('ser perigoso não nada bom','desgosto'),
('você e perigoso demais para minha filhas','desgosto'),
('que fetido este esgoto','desgosto'),
('que fedido você esta','desgosto'),
('que cachorro malcheiroso','desgosto'),
('hora que ultraje','desgosto'),
('e ultrajante da sua parte','desgosto'),
('situação desagradável essa','desgosto'),
('você só me da desgosto','desgosto'),
('tenho aversão a pessoas assim','desgosto'),
('antipatia e um mal da sociedade','desgosto'),
('que criatura abominável','desgosto'),
('e depressiva a maneira como você vê o mundo','desgosto'),
('me desagrada sua presença na festa','desgosto'),
('sinto asco dessa coisa','desgosto'),
('que hediondo!','desgosto'),
('vou golfar o cafe fora','desgosto'),
('hora que garota detestável!','desgosto'),
('estou nauseada','desgosto'),
('isso que você disse foi muito grave','desgosto'),
('não seja obsceno na frente das crianças','desgosto'),
('não seja rude com as visitas','desgosto'),
('esse assunto me da repulsa','desgosto'),
('que criança terrivelmente travessa','desgosto'),
('que criança mal educada','desgosto'),
('estou indisposta te dar o divorcio','desgosto'),
('tão patetico, não tem nada mais rude para dizer?','desgosto'),
('por motivo torpe, com emprego de meio cruel e com impossibilidade de defesa para a vítima','desgosto'),
('a inveja e tão vil e vergonhosa que ninguem se atreve a confessá-la','desgosto'),
('o miserável receio de ser sentimental e o mais vil de todos os receios modernos','desgosto'),
('travesso gato quando fica com saudades do dono mija no sapato','desgosto'),
('isso e um ato detestável e covarde','desgosto'),
('revelam apenas o que e destrutivo e detestável para o povo','desgosto'),
('não sei como e a vida de um patife, mais a de um homem honesto e abominável','desgosto'),
('há coisas que temos que suportar para não acharmos a vida insuportável','desgosto'),
('as injurias do tempo e as injustiças do homem','desgosto'),
('odioso e desumano','desgosto'),
('você não publicará conteúdo odiento, pornográfico ou ameaçador','desgosto'),
('rancoroso e reprimido','desgosto'),
('não há animal mais degradante, estúpido, covarde, lamentável, egoísta, rancoroso e invejoso do que o homem','desgosto'),
('o virulento debate ente políticos','desgosto'),

('por favor não me abandone','tristeza'),
('não quero ficar sozinha','tristeza'),
('não me deixe sozinha','tristeza'),
('estou abatida','tristeza'),
('ele esta todo abatido','tristeza'),
('tão triste suas palavras','tristeza'),
('seu amor não e mais meu ','tristeza'),
('estou aborrecida','tristeza'),
('isso vai me aborrecer ','tristeza'),
('estou com muita aflição','tristeza'),
('me aflige o modo como fala','tristeza'),
('estou em agonia com meu intimo','tristeza'),
('me sinto só','tristeza'),
('O seu amor só me trouxe lágrimas','tristeza'),
('A vida pode machucar muito mais que a morte','tristeza'),
('Está cada vez mais difícil esconder a dor','tristeza'),
('Chorar é a forma que o meu coração consegue se expressar','tristeza'),
('Decepcionado com a vida','tristeza'),
('São muitas decepções para um só coração','tristeza'),
('por favor não me abandone','tristeza'),
('não quero ficar sozinha','tristeza'),
('não me deixe sozinha','tristeza'),
('estou abatida','tristeza'),
('ele esta todo abatido','tristeza'),
('tão triste suas palavras','tristeza'),
('seu amor não e mais meu','tristeza'),
('estou aborrecida','tristeza'),
('isso vai me aborrecer','tristeza'),
('estou com muita aflição','tristeza'),
('me aflige o modo como fala','tristeza'),
('estou em agonia com meu intimo','tristeza'),
('não quero fazer nada','tristeza'),
('me sinto ansiosa e tensa','tristeza'),
('não consigo parar de chorar','tristeza'),
('não consigo segurar as lagrimas','tristeza'),
('e muita dor perder um ente querido','tristeza'),
('estou realmente arrependida','tristeza'),
('acho que o carma volta, pois agora sou eu quem sofro','tristeza'),
('você não cumpriu suas promessas','tristeza'),
('me sinto amargurada','tristeza'),
('coitado esta tão triste','tristeza'),
('já e tarde de mais','tristeza'),
('nosso amor acabou','tristeza'),
('essa noite machuca só para mim','tristeza'),
('eu não estou mais no seu coração','tristeza'),
('você mudou comigo','tristeza'),
('quando eu penso em você realmente dói','tristeza'),
('como se fosse nada você vê minhas lagrimas','tristeza'),
('você disse cruelmente que não se arrependeu','tristeza'),
('eu nunca mais vou te ver','tristeza'),
('ela esta com depressão','tristeza'),
('a depressão aflige as pessoas','tristeza'),
('estar depressivo e muito ruim','tristeza'),
('estou derrotada e deprimida depois deste dia','tristeza'),
('e comovente te ver dessa maneira','tristeza'),
('e comovente ver o que os filhos do brasil passam','tristeza'),
('como me sinto culpada','tristeza'),
('estou abatida','tristeza'),
('a ansiedade tomou conta de mim','tristeza'),
('as pessoas não gostam do meu jeito','tristeza'),
('adeus passamos bons momentos juntos','tristeza'),
('sinto sua falta','tristeza'),
('ele não gostou da minha comida','tristeza'),
('estou sem dinheiro para a comida','tristeza'),
('queria que fosse o ultimo dia da minha vida','tristeza'),
('você está com vergonha de mim','tristeza'),
('ela não aceitou a minha proposta','tristeza'),
('era o meu ultimo centavo','tristeza'),
('reprovei de ano na faculdade','tristeza'),
('afinal você só sabe me desfazer','tristeza'),
('eu falhei em tudo nessa vida','tristeza'),
('eu fui muito humilhado','tristeza'),
('e uma história muito triste','tristeza'),
('ninguem acredita em mim','tristeza'),
('eu não sirvo para nada mesmo','tristeza'),
('droga, não faço nada direito','tristeza'),
('sofrimento em dobro na minha vida','tristeza'),
('fui demitida essa semana','tristeza'),
('as crianças sofrem ainda mais que os adultos','tristeza'),
('pra mim um dia e ruim, o outro e pior','tristeza'),
('de repente perdi o apetite','tristeza'),
('oh que dia infeliz','tristeza'),
('estamos afundados em contas','tristeza'),
('nem um milagre pode nos salvar','tristeza'),
('só me resta a esperança','tristeza'),
('pior que isso não pode ficar','tristeza'),
('meu salário e baixo','tristeza'),
('não passei no vestibular','tristeza'),
('ninguem se importa comigo','tristeza'),
('ninguem lembrou do meu aniversário','tristeza'),
('tenho tanto azar','tristeza'),
('o gosto da vingança e amargo','tristeza'),
('sou uma mulher amargurada depois de que você me deixou','tristeza'),
('estou desanimada com a vida','tristeza'),
('e um desanimo só coitadinha','tristeza'),
('a derrota e depressiva','tristeza'),
('discriminar e desumano','tristeza'),
('que desanimo','tristeza'),
('e uma desonra para o pais','tristeza'),
('a preocupação deveria nos levar a ação não a depressão','tristeza'),
('passamos ao desalento e a loucura','tristeza'),
('aquele que nunca viu a tristeza nunca reconhecerá a alegria','tristeza'),
('cuidado com a tristeza ela e um vicio','tristeza'),
]
BaseTeste =[
('não precisei pagar o ingresso','alegria'),
('se eu ajeitar tudo fica bem','alegria'),
('minha fortuna ultrapassa a sua','alegria'),
('sou muito afortunado','alegria'),
('e benefico para todos esta nova medida','alegria'),
('ficou lindo','alegria'),
('achei esse sapato muito simpático','alegria'),
('estou ansiosa pela sua chegada','alegria'),
('congratulações pelo seu aniversário','alegria'),
('delicadamente ele a colocou para dormir','alegria'),
('a musica e linda','alegria'),
('sem musica eu não vivo','alegria'),
('conclui uma tarefa muito difícil','alegria'),
('Simplesmente viva a vida','alegria'),
('Chorar é necessário, mas sorrir é essencial','alegria'),
('O seu sorriso pode mudar o dia de alguém','alegria'),
('Sorria, você foi feito para isso','alegria'),
('Todas as pessoas do mundo sorriem no mesmo idioma','alegria'),


(' com consegue ser tão bela? ',' surpresa '),
(' essa e realmente uma casa deslumbrante ',' surpresa '),
(' superou minhas expectativas ',' surpresa '),
(' e admirável a maneira como se comporta ',' surpresa '),
(' isso e realmente chocante ',' surpresa '),
(' algumas noticias me surpreenderam no noticiário ',' surpresa '),
(' surpreendente sua festa ',' surpresa '),
(' estou tremendo de alegria ',' surpresa '),
(' chocou grande parte do mundo ',' surpresa '),
(' eu ficaria muito espantado com a sua vinda ',' surpresa '),
(' ele e admirável ',' surpresa '),
(' sua beleza me surpreendeu ',' surpresa '),
(' seus olhos são surpreendentemente verdes ',' surpresa '),
(' os políticos se surpreendem quando alguem acredita neles ',' surpresa '),
(' estou perplexa com essas denuncias ',' surpresa '),
(' fiquei perplexo com suas palavras ',' surpresa '),
(' estou abismado com sua prosa ',' surpresa '),
(' eu ficaria realmente abismado se me dissessem isso ',' surpresa '),
(' o grupo foi surpreendido enquanto lavava o carro ',' surpresa '),
(' estou boquiaberto com as imagens ',' surpresa '),
(' estou boquiaberto com essas suas palavras ',' surpresa '),
(' esse quadro e maravilhoso ',' surpresa '),
(' este carro me deixou maravilhado ',' surpresa '),
(' estou maravilhada ',' surpresa '),
(' essa expectativa esta me matando ',' surpresa '),
(' vou caminhar sempre na expectativa de encontrá-lo ',' surpresa '),
(' você emudece minhas palavras ',' surpresa '),
(' minhas palavras vão emudecer se não parar de me surpreender ','surpresa'),
(' a mulher e um efeito deslumbrante da natureza ',' surpresa '),
(' estou deslumbrada com essas jóias ',' surpresa '),
(' isso e romântico e deslumbrante ',' surpresa '),
(' isso pode ser surpreendentemente deslumbrante ',' surpresa '),
(' trabalho deslumbrante ',' surpresa '),
(' essas pessoas são esplêndida ',' surpresa '),
(' e esplendido como o ceu se encontra no momento ',' surpresa '),
(' e um carro fantástico ',' surpresa '),
(' um edifício realmente fantástico ',' surpresa '),

(' vou vetar o orçamento ao cliente ',' raiva '),
(' meus pais não consentiram nosso casamento ',' raiva '),
(' eu odiei este perfume ',' raiva '),
(' seu descaso e frustrante ',' raiva '),
(' me sinto completamente amarga ',' raiva '),
(' desprezo muito o seu trabalho ',' raiva '),
(' estamos descontentes por nossa família ',' raiva '),
(' vou infernizar a sua empresa ',' raiva '),
(' estou furioso com estes valores ',' raiva '),
(' obrigaram o rapaz a sair ',' raiva '),
(' como ele pode deixar de lado? ',' raiva '),
(' são apenas injurias sobre mim ',' raiva '),
(' estou enfurecido com a situação dessa empresa ',' raiva '),
(' estou com o diabo no corpo ',' raiva '),
(' isso foi diabólico ',' raiva '),
(' tenho aversão à gente chata ',' raiva '),
(' não vou perdoar sua traição ',' raiva '),
(' esse dinheiro sujo e corrupto ',' raiva '),
(' eles me crucificam o tempo todo ',' raiva '),
(' eu vou enlouquecer com todo este barulho ',' raiva '),
(' não agüento todo esse assedio ',' raiva '),
(' cólera do dragão ',' raiva '),
(' isso e ridículo! ',' raiva '),
(' da próxima vez, vou inventar tudo sozinho ',' raiva '),
(' seus tolos! deixaram ele escapar! ',' raiva '),
(' jamais te perdoarei ',' raiva '),
(' o que e isso? outra multa ',' raiva '),
(' você passou dos limites! ',' raiva '),
(' sente-se e cale a boca ',' raiva '),
(' ingratosvermesvocês me pagam! ',' raiva '),
(' saiam da dai, se não arranco vocês dai! ',' raiva '),
(' você já me causou problemas suficientes ',' raiva '),
(' isso foi a gota dagua ',' raiva '),
(' o que você tem com isso? ',' raiva '),
(' não vejo a hora de me livrar de você ',' raiva '),
(' já entendi a jogada seus safados! ',' raiva '),
(' você não merece piedade ',' raiva '),
(' saia de perto de mim ',' raiva '),
(' suma daqui, ou arranco seu couro! ',' raiva '),
(' estou revoltado com essa situação ',' raiva '),
(' seu idiota! ',' raiva '),
(' não, eu não vou te emprestar dinheiro! ',' raiva '),
(' você não passa de um cafajeste! vai embora ',' raiva '),
(' pare de frescura e vá trabalhar ',' raiva '),
(' eles merecem uma lição ',' raiva '),
(' ainda estou muito bravo com você ',' raiva '),
(' eu preciso surrar aquela chantagista ',' raiva '),
(' olha o que você fez! derramou! ',' raiva '),
(' você está pedindo pra apanhar! ',' raiva '),
(' me deixa em paz! ',' raiva '),
(' morra maldito, morra! ',' raiva '),
(' você e mais irritante de perto ',' raiva '),
(' e bom fechar o bico ',' raiva '),

('isso tudo e um erro','tristeza'),
('eu sou errada eu sou errante','tristeza'),
('tenho muito dó do cachorro','tristeza'),
('e dolorida a perda de um filho','tristeza'),
('essa tragedia vai nos abalar para sempre','tristeza'),
('perdi meus filhos','tristeza'),
('perdi meu curso','tristeza'),
('sou só uma chorona','tristeza'),
('você e um chorão','tristeza'),
('se arrependimento matasse','tristeza'),
('me sinto deslocado em sala de aula','tristeza'),
('foi uma passagem fúnebre','tristeza'),
('nossa condolências e tristeza a sua perda','tristeza'),
('Meu frágil coração cansou de te esperar','tristeza'),
('Minha ferida pode até sarar, mas sei que ela me deixará marcas','tristeza'),
('Nada nos deixa tão vulneráveis quanto a decepção','tristeza'),
('A pior solidão é estar com alguém e ainda assim se sentir solitário','tristeza'),

('este lugar e mal assombrado','medo'),
('estou assombrado pela crise financeira','medo'),
('mesmo aterrorizado lembro de você','medo'),
('aterrorizado e suando frio','medo'),
('um grupo de elefantes selvagens tem aterrorizado vilas','medo'),
('me sinto intimidada pela sua presença','medo'),
('tenho medo de ser advertida novamente','medo'),
('estou correndo o risco de ser advertido','medo'),
('estou correndo riscos de saúde','medo'),
('os riscos são reais','medo'),
('podemos perder muito dinheiro com essa investida','medo'),
('socorro, fui intimado a depor','medo'),
('fui notificado e estou com medo de perde a guarda da minha filha','medo'),
('estou angustiada com meus filhos na rua','medo'),
('e abominável o que fazem com os animais','medo'),
('foi terrível o tigre quase o matou','medo'),
('me advertiram sobre isso','medo'),
('eu me assustei com você','medo'),

('o mundo e feio como o pecado','desgosto'),
('a coisa mais difícil de esconder e aquilo que não existe','desgosto'),
('você errou feio aquele gol','desgosto'),
('nunca vou me casar sou muito feia','desgosto'),
('os golpes da adversidade são terrivelmente amargos','desgosto'),
('os homem ficam terrivelmente chatos','desgosto'),
('abominavelmente convencido','desgosto'),
('terrivelmente irritado','desgosto'),
('as instituições publicas estão terrivelmente decadentes','desgosto'),
('a população viveu em isolamento por muito tempo','desgosto'),
('estou terrivelmente preocupada','desgosto'),
('o nacionalismo e uma doença infantil','desgosto'),
('se me es antipático a minha negação esta pronta','desgosto'),
('muitos documentários sobre esse casal antipático','desgosto'),
('sua beleza não desfaça sua antipatia','desgosto'),
('esta e uma experiência desagradável','desgosto'),
('desagradável estrago nos banheiros','desgosto'),
('o mais irritante no amor e que se trata de um crime que precisa de um cúmplice','desgosto'),
('a situação nos causa grande incomodo','desgosto'),
('estou preocupado com o incomodo na garganta','desgosto'),
('simplesmente não quero amolação da policia','desgosto'),
('você e uma criaturinha muito impertinente','desgosto'),
('o peso e a dor da vida','desgosto'),
('me arrependo amargamente de minhas ações','desgosto'),
('o destino e cruel e os homens não são dignos de compaixão','desgosto'),
('o ódio conduz ao isolamento cruel e ao desespero','desgosto'),
('encerrou com o massacre mais repudiável e asqueroso que se conhece','desgosto'),
('de mal gosto e asqueroso','desgosto'),
('tudo e inserto neste mundo hediondo','desgosto'),
('o crime de corrupção e um crime hediondo','desgosto'),
('o rio esta fetido e de cor escura','desgosto'),
('muito lixo no rio o deixa malcheiroso','desgosto'),
('existe uma laranja podre no grupo e já desconfiamos quem e','desgosto'),
('foi de repente estou machucado e me sentindo enjoado','desgosto'),
('eu fiquei enojado','desgosto'),
('daqui alguns meses vou embora deste pais que já estou nauseado','desgosto')
]

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
    for(palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemmerTreinamento = aplicastemmer(BaseTreinamento)
frasescomstemmerTeste = aplicastemmer(BaseTeste)

def buscaPalavras(frases):
    todasPalavras = []
    for(palavras, emocao) in frases:
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

teste = 'eu amo voçê'
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

print("\nPorcentagem do classificador de Naive Bayes do Teste Realizado:\n")
for classe in distribuicao.samples():
   print("%s: %.5f" % (classe, distribuicao.prob(classe)))

print("\nSentenças do Tokenize do Teste Realizado:\n")
tokens = word_tokenize(teste)
tags = pos_tag(tokens)

for token, tag in zip (tokens,tags):
   print(token,tag)
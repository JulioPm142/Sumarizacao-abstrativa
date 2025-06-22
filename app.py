from transformers import pipeline

#fonte:https://pdfs.semanticscholar.org/8f18/27682aba66e51cd8a8329bd3496f7c76591c.pdf

def summarize_text(input_text, max_length=250, min_length=150):
    # modelo de sumarização em português
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    
    # Realiza a sumarização
    summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    
    # Retorna o texto sumarizado
    return summary[0]['summary_text']

# Texto de entrada
input_text = """
A Inteligência Artificial (IA) é um tema que tem ganhado cada vez mais destaque na
educação, principalmente na modalidade a distância. A IA pode ser definida como
um conjunto de algoritmos e técnicas que permitem que as máquinas aprendam a partir de
dados e experiências anteriores, e possam tomar decisões de forma autônoma.
Nesse contexto, é possível observar as vantagens da IA na educação, como a personalização
do ensino, a possibilidade de feedback imediato, a acessibilidade a conteúdos de qualidade
e a melhoria do processo de aprendizagem. A personalização do ensino, por exemplo, é um
aspecto muito importante, pois cada aluno possui necessidades e habilidades específicas. Com
a IA, é possível adaptar o ensino às características de cada estudante, tornando o processo de
aprendizagem mais eficiente e significativo.
Por outro lado, a IA também apresenta desafios e desvantagens para a educação. Um dos
principais desafios é a atualização constante dos sistemas, já que a tecnologia evolui rapidamente
e é necessário acompanhar essas mudanças para que a IA possa ser efetivamente aplicada na
educação. Além disso, há a preocupação em garantir a privacidade e segurança dos dados dos
estudantes, bem como a possibilidade de discriminação algorítmica.
A aplicação prática da IA na educação pode ser vista em diversos exemplos bemsucedidos. O Watson Education da IBM, por exemplo, é uma plataforma que oferece suporte
à aprendizagem personalizada e colaborativa, permitindo que os alunos trabalhem em projetos
interativos e tenham acesso a feedback imediato. Essa plataforma pode ajudar os professores a
identificar lacunas no conhecimento dos alunos e fornecer intervenções personalizadas.
Entretanto, a implementação efetiva da IA na educação envolve desafios tanto para os
docentes quanto para os estudantes. Os professores precisam se adaptar às novas tecnologias e
aprender a utilizar as ferramentas de IA de forma eficiente, além de estar sempre atualizados
em relação às mudanças na tecnologia. Já os estudantes precisam ser treinados para utilizar as
ferramentas de IA, e devem estar preparados para lidar com as mudanças na forma de ensino.
Diante dessas considerações, é possível afirmar que a IA pode trazer muitos benefícios
para a educação, principalmente na modalidade a distância. No entanto, é preciso estar ciente
dos desafios e desvantagens envolvidos na aplicação da IA na educação, e estar preparado para
lidar com eles. A IA pode ajudar a tornar o processo de ensino e aprendizagem eficientes, desde
que seja aplicada de forma adequada e consciente.
"""


summary = summarize_text(input_text)


print("Sumarização abstrativa:")
print(summary)
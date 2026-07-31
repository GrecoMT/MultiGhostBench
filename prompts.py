#-------------outline
def get_prompt_out_en(NUM_WORDS, GENRES, CENTURY):
    ret = f"""Generate a detailed outline for a {NUM_WORDS}-word novel in your own style based on these genres: {GENRES} set in the {CENTURY} century. Include word counts and major plot points. 

    Please insert the generated text between <START>...<END> tags. Any other message or metadata outside these tags."""
    return ret 

def get_prompt_out_it(NUM_WORDS, GENRES, CENTURY):
    ret = f"""Genera una struttura dettagliata per un romanzo di {NUM_WORDS} parole nel tuo stile basata su questi generi: {GENRES} ambientata nel {CENTURY} secolo. Includi il numero di parole e i principali punti della trama.

    Per favore inserisci il testo generato tra i tag <START>...<END>. Qualsiasi altro messaggio o metadato al di fuori di questi tag non è consentito."""
    return ret

def get_prompt_out_es(NUM_WORDS, GENRES, CENTURY ):
    ret = f"""Genere un esquema detallado para una novela de {NUM_WORDS} palabras en su propio estilo, basada en los siguientes géneros: {GENRES}, ambientada en el siglo {CENTURY}. Incluya el recuento de palabras y los principales puntos de la trama. 

    Inserte el texto generado entre las etiquetas <START>...<END>. Cualquier otro mensaje o metadato fuera de estas etiquetas está prohibido."""
    return ret

def get_prompt_out_de(NUM_WORDS, GENRES, CENTURY):
    ret = f"""Erstellen Sie eine detaillierte Gliederung für einen Roman mit {NUM_WORDS} Wörtern in Ihrem eigenen Stil, basierend auf folgenden Genres: {GENRES}, angesiedelt im {CENTURY}. Jahrhundert. Fügen Sie Wortanzahlen und die wichtigsten Handlungspunkte hinzu. 
    
    Bitte fügen Sie den generierten Text zwischen den Tags <START>...<END> ein. Jegliche andere Nachricht oder Metadaten außerhalb dieser Tags sind nicht erlaubt."""
    return ret

def get_prompt_out_ch(NUM_WORDS, GENRES, CENTURY):
    ret = f"""根据以下类型：{GENRES}，并以{CENTURY}世纪为背景，请用你自己的风格生成一个总字数为{NUM_WORDS}字的小说的详细大纲。请包含每个部分的字数以及主要情节要点。

    请将生成的文本放在 <START>...<END> 标签之间。任何其他消息或元数据都不允许出现在这些标签之外。"""
    return ret

def get_prompt_out_ru(NUM_WORDS, GENRES, CENTURY):
    ret = f"""Сгенерируй подробный план романа объёмом {NUM_WORDS} слов в собственном стиле на основе следующих жанров: {GENRES}, действие которого происходит в {CENTURY} веке. Включи количество слов и основные сюжетные точки.

    Пожалуйста, помести сгенерированный текст между тегами <START>...<END>. Любое другое сообщение или метаданные вне этих тегов."""
    return ret

PROMPT_BUILDERS_OUTLINE = {
    "en": get_prompt_out_en,
    "it": get_prompt_out_it,
    "es": get_prompt_out_es,
    "de": get_prompt_out_de,
    "ch": get_prompt_out_ch,
    "ru": get_prompt_out_ru
}

#-------------summary
def get_prompt_sum_en(start, text, summary_till_now):
    ret_start = f"""Below is the beginning part of a story:

    ---

    {text}

    ---

    We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot. Write a summary for the excerpt provided above, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations. You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative. Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go. The summary could include multiple paragraphs.

    Summary:"""
    ret_nstart = f"""Below is a segment from a story:

    ---

    {text}

    ---

    Below is a summary of the story up until this point:

    ---

    {summary_till_now}

    ---

    We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot. You are required to update the summary to incorporate any new vital information in the current excerpt. This information may relate to key events, backgrounds, settings, characters, their objectives, and motivations. You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative. Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go. The updated summary could include multiple paragraphs.

    Updated summary:"""
    return ret_start if start else ret_nstart

def get_prompt_sum_it(start, text, summary_till_now):
    ret_start = f"""Di seguito è riportata la parte iniziale di una storia:

    ---

    {text}

    ---

    Stiamo esaminando i segmenti di una storia in modo sequenziale per aggiornare progressivamente un'unica sintesi completa dell'intera trama. Scrivi un riassunto dell'estratto fornito sopra, assicurandoti di includere le informazioni essenziali relative agli eventi chiave, ai retroscena, alle ambientazioni, ai personaggi, ai loro obiettivi e alle loro motivazioni. Devi introdurre brevemente personaggi, luoghi e altri elementi principali qualora vengano menzionati per la prima volta nel riassunto. La storia potrebbe presentare una narrazione non lineare, flashback, passaggi tra mondi alternativi o cambi di punto di vista. Pertanto, organizza il riassunto in modo da offrire una narrazione coerente e cronologicamente ordinata. Nonostante questo processo di aggiornamento incrementale, il riassunto deve apparire come se fosse stato scritto in un'unica soluzione. Il testo può essere articolato in più paragrafi.

    Riassunto:"""
    ret_nstart = f"""Di seguito è riportato un segmento di una storia:

    ---

    {text}

    ---

    Di seguito è riportato il riassunto della storia fino ad ora: 

    ---

    {summary_till_now}

    ---

    Stiamo esaminando i segmenti di una storia in modo sequenziale per aggiornare progressivamente un'unica sintesi completa dell'intera trama. Ti è richiesto di aggiornare il riassunto in modo da incorporare eventuali nuove informazioni essenziali presenti nell'estratto corrente. Tali informazioni possono riguardare eventi chiave, retroscena, ambientazioni, personaggi, i loro obiettivi e le loro motivazioni. Devi introdurre brevemente personaggi, luoghi e altri elementi principali qualora vengano menzionati per la prima volta nel riassunto. La storia può presentare una narrazione non lineare, flashback, passaggi tra mondi alternativi o cambi di punto di vista. Pertanto, organizza il riassunto in modo da offrire una narrazione coerente e cronologicamente ordinata. Nonostante questo processo di aggiornamento graduale, il riassunto deve apparire come se fosse stato scritto in un'unica soluzione. Il riassunto aggiornato può essere articolato in più paragrafi.

    Riassunto aggiornato:"""
    return ret_start if start else ret_nstart


    ret_start = f"""Ci-dessous se trouve le début d’une histoire :

    ---

    {text}

    ---

    Nous examinons les segments d’une histoire séquentiellement afin de mettre à jour progressivement un résumé complet de l’intrigue. Rédigez un résumé de l’extrait ci-dessus en incluant les informations essentielles concernant les événements clés, les contextes, les décors, les personnages, leurs objectifs et leurs motivations. Vous devez brièvement introduire les personnages, les lieux et les autres éléments majeurs s’ils sont mentionnés pour la première fois dans le résumé. L’histoire peut comporter des récits non linéaires, des retours en arrière, des passages entre mondes alternatifs ou différents points de vue, etc. Vous devez donc organiser le résumé de manière cohérente et chronologique. Malgré ce processus étape par étape, le résumé doit sembler rédigé en une seule fois. Il peut comporter plusieurs paragraphes.

    Résumé :"""
    ret_nstart = f"""Ci-dessous se trouve un segment d’une histoire :

    ---

    {text}

    ---

    Voici un résumé de l’histoire jusqu’à présent :

    ---

    {summary_till_now}

    ---

    Nous examinons les segments d’une histoire séquentiellement afin de mettre à jour progressivement un résumé complet de l’intrigue. Vous devez mettre à jour le résumé en intégrant toute nouvelle information essentielle contenue dans l’extrait actuel. Cela peut concerner des événements clés, des contextes, des décors, des personnages, leurs objectifs et leurs motivations. Vous devez brièvement introduire les personnages, les lieux et les autres éléments majeurs s’ils apparaissent pour la première fois. L’histoire peut comporter des récits non linéaires, des retours en arrière, des passages entre mondes alternatifs ou différents points de vue, etc. Vous devez donc organiser le résumé de manière cohérente et chronologique. Malgré ce processus étape par étape, le résumé mis à jour doit sembler rédigé en une seule fois. Il peut comporter plusieurs paragraphes.

    Résumé mis à jour :"""
    return ret_start if start else ret_nstart

def get_prompt_sum_es(start, text, summary_till_now):
    ret_start = f"""A continuación se presenta el comienzo de una historia:

    ---

    {text}

    ---

    Estamos revisando segmentos de una historia de forma secuencial para actualizar gradualmente un resumen completo de toda la trama. Escriba un resumen del fragmento anterior incluyendo información esencial sobre eventos clave, antecedentes, escenarios, personajes, sus objetivos y motivaciones. Debe introducir brevemente a los personajes, lugares y otros elementos importantes si se mencionan por primera vez. La historia puede incluir narrativas no lineales, flashbacks, cambios entre mundos alternativos o puntos de vista distintos, etc. Por lo tanto, organice el resumen de manera coherente y cronológica. A pesar de este proceso paso a paso, el resumen debe parecer escrito de una sola vez. Puede incluir varios párrafos.

    Resumen:"""
    ret_nstart = f"""A continuación se presenta un segmento de una historia:

    ---

    {text}

    ---

    A continuación se muestra un resumen de la historia hasta este punto:

    ---

    {summary_till_now}

    ---

    Estamos revisando segmentos de una historia de forma secuencial para actualizar gradualmente un resumen completo de toda la trama. Debe actualizar el resumen incorporando cualquier nueva información esencial presente en el fragmento actual. Esta información puede referirse a eventos clave, antecedentes, escenarios, personajes, sus objetivos y motivaciones. Debe introducir brevemente personajes, lugares y otros elementos importantes si aparecen por primera vez. La historia puede incluir narrativas no lineales, flashbacks, cambios entre mundos alternativos o puntos de vista distintos, etc. Por lo tanto, organice el resumen de manera coherente y cronológica. A pesar de este proceso paso a paso, el resumen actualizado debe parecer escrito de una sola vez. Puede incluir varios párrafos.

    Resumen actualizado:"""
    return ret_start if start else ret_nstart

def get_prompt_sum_de(start, text, summary_till_now):
    ret_start = f"""Im Folgenden finden Sie den Anfang einer Geschichte:

    ---

    {text}

    ---

    Wir gehen die Segmente einer Geschichte nacheinander durch, um schrittweise eine umfassende Zusammenfassung der gesamten Handlung zu erstellen. Schreiben Sie eine Zusammenfassung des obigen Auszugs und berücksichtigen Sie dabei wesentliche Informationen zu wichtigen Ereignissen, Hintergründen, Schauplätzen, Figuren, deren Zielen und Motivationen. Führen Sie Figuren, Orte und andere zentrale Elemente kurz ein, falls sie erstmals erwähnt werden. Die Geschichte kann nicht-lineare Erzählstrukturen, Rückblenden, Wechsel zwischen alternativen Welten oder Perspektiven usw. enthalten. Ordnen Sie die Zusammenfassung daher kohärent und chronologisch. Trotz dieses schrittweisen Prozesses soll die Zusammenfassung wie aus einem Guss wirken. Sie kann mehrere Absätze umfassen.

    Zusammenfassung:"""
    ret_nstart = f"""Im Folgenden finden Sie einen Abschnitt einer Geschichte:

    ---

    {text}

    ---

    Hier ist eine Zusammenfassung der Geschichte bis zu diesem Punkt:

    ---

    {summary_till_now}

    ---

    Wir gehen die Segmente einer Geschichte nacheinander durch, um schrittweise eine umfassende Zusammenfassung der gesamten Handlung zu erstellen. Sie müssen die bestehende Zusammenfassung aktualisieren und neue wesentliche Informationen aus dem aktuellen Abschnitt integrieren. Diese können wichtige Ereignisse, Hintergründe, Schauplätze, Figuren, deren Ziele und Motivationen betreffen. Führen Sie Figuren, Orte und andere zentrale Elemente kurz ein, falls sie erstmals erscheinen. Die Geschichte kann nicht-lineare Erzählstrukturen, Rückblenden, Wechsel zwischen alternativen Welten oder Perspektiven usw. enthalten. Ordnen Sie die aktualisierte Zusammenfassung daher kohärent und chronologisch. Trotz dieses schrittweisen Prozesses soll sie wie aus einem Guss wirken. Sie kann mehrere Absätze umfassen.

    Aktualisierte Zusammenfassung:"""
    return ret_start if start else ret_nstart

def get_prompt_sum_ch(start, text, summary_till_now):
    ret_start = f"""下面是一个故事的开头部分：

    ---

    {text}

    ---

    我们将按顺序逐段阅读这个故事，以逐步更新整个情节的综合摘要。请为上面提供的片段撰写一个摘要，并确保包含与关键事件、背景、场景、人物及其目标和动机相关的重要信息。如果人物、地点或其他主要元素在摘要中首次出现，你必须对其进行简要介绍。故事可能包含非线性叙事、闪回、不同世界或视角之间的切换等。因此，你需要将摘要组织为一个一致且按时间顺序展开的叙述。尽管这是一个逐步更新摘要的过程，你仍然需要生成一个看起来像一次性写成的完整摘要。摘要可以包含多个段落。

    摘要："""
        
    ret_nstart = f"""下面是一个故事中的一段内容：

    ---

    {text}

    ---

    下面是截至目前为止的故事摘要：

    ---

    {summary_till_now}

    ---

    我们将按顺序逐段阅读这个故事，以逐步更新整个情节的综合摘要。你需要更新该摘要，以纳入当前片段中的任何新的重要信息。这些信息可能涉及关键事件、背景、场景、人物及其目标和动机。如果人物、地点或其他主要元素在摘要中首次出现，你必须对其进行简要介绍。故事可能包含非线性叙事、闪回、不同世界或视角之间的切换等。因此，你需要将摘要组织为一个一致且按时间顺序展开的叙述。尽管这是一个逐步更新摘要的过程，你仍然需要生成一个看起来像一次性写成的完整摘要。更新后的摘要可以包含多个段落。

    更新后的摘要："""
    
    return ret_start if start else ret_nstart

def get_prompt_sum_ru(start, text, summary_till_now):
    ret_start = f"""Ниже приведена начальная часть истории:

    ---

    {text}

    ---

    Мы последовательно рассматриваем сегменты истории, чтобы постепенно обновлять одно всеобъемлющее краткое содержание всего сюжета. Напиши краткое содержание для приведённого выше фрагмента, обязательно включив важную информацию, связанную с ключевыми событиями, предысториями, местами действия, персонажами, их целями и мотивациями. Ты должен кратко представить персонажей, места и другие основные элементы, если они впервые упоминаются в кратком содержании. История может включать нелинейное повествование, флэшбэки, переходы между альтернативными мирами или точками зрения и т. д. Поэтому ты должен организовать краткое содержание так, чтобы оно представляло последовательное и хронологическое повествование. Несмотря на этот пошаговый процесс обновления краткого содержания, тебе нужно создать краткое содержание, которое выглядит так, будто оно написано за один раз. Краткое содержание может включать несколько абзацев.

    Краткое содержание:"""
    ret_nstart = f"""Ниже приведён сегмент истории:

    ---

    {text}

    ---

    Ниже приведено краткое содержание истории до этого момента:

    ---

    {summary_till_now}

    ---

    Мы последовательно рассматриваем сегменты истории, чтобы постепенно обновлять одно всеобъемлющее краткое содержание всего сюжета. Тебе нужно обновить краткое содержание, чтобы включить любую новую важную информацию из текущего фрагмента. Эта информация может относиться к ключевым событиям, предысториям, местам действия, персонажам, их целям и мотивациям. Ты должен кратко представить персонажей, места и другие основные элементы, если они впервые упоминаются в кратком содержании. История может включать нелинейное повествование, флэшбэки, переходы между альтернативными мирами или точками зрения и т. д. Поэтому ты должен организовать краткое содержание так, чтобы оно представляло последовательное и хронологическое повествование. Несмотря на этот пошаговый процесс обновления краткого содержания, тебе нужно создать краткое содержание, которое выглядит так, будто оно написано за один раз. Обновлённое краткое содержание может включать несколько абзацев.

    Обновлённое краткое содержание:"""
    return ret_start if start else ret_nstart

PROMPT_CREATE_SUM = {
    "en": get_prompt_sum_en,
    "it" : get_prompt_sum_it,
    "es" : get_prompt_sum_es,
    "de" : get_prompt_sum_de,
    "ch" : get_prompt_sum_ch,
    "ru" : get_prompt_sum_ru
}

def get_prompt_comprsum_en(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""Below is a summary of part of a story:

    ---

    {summary}

    ---

    Currently, this summary contains {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} words. Your task is to condense it to less than {MAX_SUMMARY_WORDS} words. The condensed summary should remain clear, overarching, and fluid while being brief. Whenever feasible, maintain details about key events, backgrounds, settings, characters, their objectives, and motivations - but express these elements more succinctly. Make sure to provide a brief introduction to characters, places, and other major components during their first mention in the condensed summary. Remove insignificant details that do not add much to the overall story line. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative.

    Condensed summary (to be within {MAX_SUMMARY_WORDS} words):"""
    return ret

def get_prompt_comprsum_it(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""Di seguito è riportato il riassunto di una parte di una storia:

    ---

    {summary}

    ---

    Al momento, questo riassunto contiene {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} parole. Il tuo compito è condensarlo in meno di {MAX_SUMMARY_WORDS} parole. Il riassunto condensato deve rimanere chiaro, generale e fluido pur essendo breve. Quando possibile, mantieni tutti i dettagli circa eventi chiavi, sfondi, luoghi e personaggi, i loro obiettivi e motivazioni - ma esprimi questi elementi più succintamente. Rimuovi dettagli insignificati che non aggiungono molto alla storia. La storia può presentare una narrazione non lineare, flashback, passaggi tra mondi alternativi o cambi di punto di vista. Quindi, dovresti organizzare il riassunto in modo che presenti una narrazione coerentee cronologica.

    Riassunto condensato (entro {MAX_SUMMARY_WORDS} parole):"""
    return ret

def get_prompt_comprsum_es(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""A continuación se muestra un resumen de parte de una historia:

    ---

    {summary}

    ---

    Actualmente, este resumen contiene {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} palabras. Tu tarea es condensarlo a menos de {MAX_SUMMARY_WORDS} palabras. El resumen condensado debe seguir siendo claro, general y fluido, al mismo tiempo que breve. Siempre que sea posible, mantén los detalles sobre eventos clave, antecedentes, escenarios, personajes, sus objetivos y motivaciones, pero expresa estos elementos de forma más sucinta. Asegúrate de proporcionar una breve introducción a los personajes, lugares y otros componentes principales cuando se mencionen por primera vez en el resumen condensado. Elimina los detalles insignificantes que no aporten mucho a la línea argumental general. La historia puede incluir narrativas no lineales, flashbacks, cambios entre mundos alternativos o puntos de vista, etc. Por lo tanto, debes organizar el resumen de modo que presente una narrativa consistente y cronológica.

    Resumen condensado (debe estar dentro de {MAX_SUMMARY_WORDS} palabras):"""
    return ret

def get_prompt_comprsum_de(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""Im Folgenden finden Sie eine Zusammenfassung eines Teils einer Geschichte:

    ---

    {summary}

    ---

    Derzeit enthält diese Zusammenfassung {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} Wörter. Ihre Aufgabe besteht darin, sie auf weniger als {MAX_SUMMARY_WORDS} Wörter zu kürzen. Die gekürzte Zusammenfassung soll klar, übergreifend und flüssig bleiben und zugleich prägnant sein. Bewahren Sie nach Möglichkeit Details zu wichtigen Ereignissen, Hintergründen, Schauplätzen, Figuren sowie deren Zielen und Motivationen, formulieren Sie diese jedoch knapper. Stellen Sie sicher, dass Figuren, Orte und andere zentrale Elemente bei ihrer ersten Erwähnung kurz eingeführt werden. Entfernen Sie unwichtige Details, die keinen wesentlichen Beitrag zur Gesamthandlung leisten. Die Geschichte kann nicht-lineare Erzählstrukturen, Rückblenden, Wechsel zwischen alternativen Welten oder Perspektiven usw. enthalten. Ordnen Sie die Zusammenfassung daher kohärent und chronologisch.

    Gekürzte Zusammenfassung (maximal {MAX_SUMMARY_WORDS} Wörter):"""
    return ret

def get_prompt_comprsum_ch(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""下面是一个故事部分内容的摘要：

    ---

    {summary}

    ---

    当前这个摘要包含 {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} 个词。你的任务是将其压缩到少于 {MAX_SUMMARY_WORDS} 个词。压缩后的摘要应当保持清晰、整体连贯和流畅，同时更加简洁。在可行的情况下，请保留关于关键事件、背景、场景、人物及其目标和动机的细节，但要用更简练的方式表达这些内容。在压缩摘要中，当人物、地点或其他重要元素第一次出现时，请对其进行简要介绍。删除对整体故事情节贡献不大的次要细节。故事可能包含非线性叙事、闪回、不同世界或视角之间的切换等。因此，你需要将摘要组织为一个一致且按时间顺序展开的叙述。

    压缩后的摘要（需控制在 {MAX_SUMMARY_WORDS} 个词以内）："""
    return ret

def get_prompt_comprsum_ru(summary, MAX_SUMMARY_WORDS, tokenizer):
    ret = f"""Ниже приведено краткое содержание части истории:

    ---

    {summary}

    ---

    В настоящее время это краткое содержание содержит {len(tokenizer.encode(summary, add_special_tokens=False))/1.5} слов. Твоя задача — сжать его до менее чем {MAX_SUMMARY_WORDS} слов. Сжатое краткое содержание должно оставаться ясным, общим и плавным, при этом быть кратким. По возможности сохраняй детали о ключевых событиях, предысториях, местах действия, персонажах, их целях и мотивациях — но выражай эти элементы более сжато. Обязательно кратко представь персонажей, места и другие основные компоненты при их первом упоминании в сжатом кратком содержании. Удали незначительные детали, которые мало что добавляют к общей сюжетной линии. История может включать нелинейное повествование, флэшбэки, переходы между альтернативными мирами или точками зрения и т. д. Поэтому ты должен организовать краткое содержание так, чтобы оно представляло последовательное и хронологическое повествование.

    Сжатое краткое содержание (не более {MAX_SUMMARY_WORDS} слов):"""
    return ret

PROMPT_COMPRESS_SUM = {
    "en" : get_prompt_comprsum_en,
    "it" : get_prompt_comprsum_it,
    "es" : get_prompt_comprsum_es,
    "de" : get_prompt_comprsum_de,
    "ch" : get_prompt_comprsum_ch,
    "ru" : get_prompt_comprsum_ru
}


#-------------novel
def get_prompt_novel_en(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""GOAL: Write a {total_words} words novel in your own style based on these genres: {genres}. The novel is set in the {century} century. We are generating the complete novel sequentially and in parts. 

    TASK: Begin writing the novel by generating the initial segment (~{NOVEL_SEGMENT_WORDS} words). Expand the narrative comprehensively, including all relevant details, while strictly adhering to the word limit specified in the outline. Do not rush through any milestone; it is acceptable to split the content across multiple parts if needed to respect the word limits. Please enclose the generated text strictly between <START> and <END> tags. Any other message or metadata outside these tags. 

    Please read the following text carefully:
    Follow the given novel outline strictly (including word counts and major plot points).

    OUTLINE:
    {outline}"""
    ret_nstart = f"""GOAL: Write a {total_words} words novel in your own style based on these genres: {genres}. The novel is set in the {century} century. We are generating the complete novel sequentially and in parts.

    TASK: You are continuing the novel generation process. Generate the next segment (~{NOVEL_SEGMENT_WORDS} words) of the novel. Expand the narrative comprehensively, including all relevant details, while strictly adhering to the word limit specified in the outline. Do not rush through any milestone; it is acceptable to split the content across multiple parts if needed to respect the word limits. Please enclose the generated text strictly between <START> and <END> tags. If novel is complete, add "END OF NOVEL" at the end of the generated text. Any other message or metadata outside these tags.

    INSTRUCTIONS:
    1. Follow the novel outline strictly, including word counts and major plot points.
    2. Maintain full continuity with the previous segment and the overall summary generated so far.
    3. Use the previous segments and summary as reference for style, tone, and story consistency.

    OUTLINE:
    {outline}

    ENDING OF PREVIOUS SEGMENT:
    {last_segment}

    SUMMARY OF NOVEL TILL NOW:
    {summary}"""
    return ret_start if start else ret_nstart

def get_prompt_novel_it(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""OBIETTIVO: Scrivi un romanzo di {total_words} parole nel tuo stile basato su questi generi: {genres}. Il romanzo deve essere ambientato nel {century} secolo. Stiamo generando il completo romanzo in modo sequenziale e in parti. 

    COMPITO: Inizia a scrivere il romanzo scrivendo il segmento iniziale (~{NOVEL_SEGMENT_WORDS} parole). Espandi la narrazione in modo comprensivo, includendo tutti i dettagli rilevanti, rimanendo aderente al limite di parole specificato nella struttura. Non affrettare lo sviluppo di nessuna tappa fondamentale; va bene dividere il contenuto in parti multiple se necessario per rispettare il limite sul numero di parole. Il testo deve essere racchiuso tra i tag <START> e <END>.  Qualsiasi altro messaggio o metadato al di fuori di questi tag non è consentito.

    Leggi il seguente testo con calma: 
    Segui la struttura del romanzo rigorosamente (anche il numero di parole e i principali punti di trama).

    STRUTTURA:
    {outline}"""

    ret_nstart = f"""OBIETTIVO: Scrivi un romanzo di {total_words} parole nel tuo stile basato su questi generi: {genres}. Il romanzo deve essere ambientato nel {century} secolo. Stiamo generando il completo romanzo in modo sequenziale e in parti. 

    COMPITO: Stai continuando il processo di generazione del romanzo. Genera il prossimo segmento (~{NOVEL_SEGMENT_WORDS} parole) del romanzo . Espandi la narrazione in modo comprensivo, includendo tutti i dettagli rilevanti, rimanendo aderente al limite di parole specificato nella struttura. Non affrettare lo sviluppo di nessuna tappa fondamentale; va bene dividere il contenuto in parti multiple se necessario per rispettare il limite sul numero di parole. Il testo deve essere racchiuso tra i tag <START> e <END>. Se il romanzo è completo, aggiungi "END OF NOVEL" alla fine del testo generato. Qualsiasi altro messaggio o metadato al di fuori di questi tag non è consentito.

    ISTRUZIONI:

    1. Segui la struttura del romanzo rigorosamente, anche il numero di parole e i punti principali della trama.
    2. Mantieni la continuità con il segmento precedente e il riassunto generato fino ad ora.
    3. Usa i segmenti precedenti e il riassunto come esempio per stile, tono e consistenza della storia.

    STRUTTURA:
    {outline}

    FINE DEL SEGMENTO PRECEDENTE:
    {last_segment}

    SUMMARY OF NOVEL TILL NOW:
    {summary}"""
    return ret_start if start else ret_nstart


def get_prompt_novel_es(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""OBJETIVO: Escribir una novela de {total_words} palabras en su propio estilo basada en los siguientes géneros: {genres}. La novela está ambientada en el siglo {century}. Generamos la novela completa de forma secuencial y por partes.

    TAREA: Comience escribiendo el segmento inicial (~{NOVEL_SEGMENT_WORDS} palabras). Desarrolle la narrativa de forma completa incluyendo todos los detalles relevantes, respetando estrictamente los límites de palabras del esquema. No apresure ningún hito; es aceptable dividir el contenido en varias partes si es necesario. Encierre estrictamente el texto generado entre <START> y <END>. Ningún otro mensaje o metadato fuera de estas etiquetas.

    ESQUEMA:
    {outline}"""
    ret_nstart = f"""OBJETIVO: Escribir una novela de {total_words} palabras en su propio estilo basada en los siguientes géneros: {genres}. Ambientada en el siglo {century}.

    TAREA: Continúe el proceso generando el siguiente segmento (~{NOVEL_SEGMENT_WORDS} palabras). Respete estrictamente los límites de palabras y mantenga coherencia total. Encierre el texto entre <START> y <END>. Si la novela termina, agregue "END OF NOVEL". Ningún otro mensaje fuera de estas etiquetas.

    INSTRUCCIONES:
    1. Siga estrictamente el esquema.
    2. Mantenga continuidad con el segmento anterior y el resumen.
    3. Utilice los segmentos y el resumen anteriores como ejemplo del estilo, el tono y la textura de la historia.

    ESQUEMA:
    {outline}

    FINAL DEL SEGMENTO ANTERIOR:
    {last_segment}

    RESUMEN HASTA AHORA:
    {summary}"""
    return ret_start if start else ret_nstart

def get_prompt_novel_de(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""ZIEL: Schreiben Sie einen Roman mit {total_words} Wörtern in Ihrem eigenen Stil basierend auf folgenden Genres: {genres}. Der Roman spielt im {century}. Jahrhundert. Die Generierung erfolgt schrittweise und segmentweise.

    AUFGABE: Beginnen Sie mit dem ersten Abschnitt (~{NOVEL_SEGMENT_WORDS} Wörter). Entwickeln Sie die Erzählung umfassend und halten Sie sich strikt an die Wortvorgaben des Plans. Text ausschließlich zwischen <START> und <END> einfügen.

    GLIEDERUNG:
    {outline}"""
    
    ret_nstart = f"""ZIEL: Fortsetzung der Romanerstellung ({total_words} Wörter insgesamt).

    AUFGABE: Generieren Sie den nächsten Abschnitt (~{NOVEL_SEGMENT_WORDS} Wörter). Strikte Einhaltung der Gliederung und Wortanzahl. Text nur zwischen <START> und <END>. Falls abgeschlossen, "END OF NOVEL" hinzufügen.

    ANWEISUNGEN:
    1. Strikte Einhaltung der Struktur, einschließlich Wortanzahl und Haupthandlungspunkte.
    2. Stellen Sie die Kontinuität zum vorherigen Abschnitt und der bisher erstellten Zusammenfassung sicher.
    3. Stilistische Konsistenz.

    GLIEDERUNG:
    {outline}

    VORHERIGER ABSCHNITT:
    {last_segment}

    BISHERIGE ZUSAMMENFASSUNG:
    {summary}"""
    return ret_start if start else ret_nstart

def get_prompt_novel_ch(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""目标：根据以下类型 {genres}，以{century}世纪为背景，用你自己的风格创作一部总字数为 {total_words} 字的小说。小说将按顺序分段生成。

    任务：开始写作小说，生成第一段内容（约 {NOVEL_SEGMENT_WORDS} 字）。在叙事中充分展开所有相关细节，并严格遵循大纲中规定的字数限制。不要匆忙推进任何情节节点；如果需要，可以将内容分成多个部分，以遵守字数限制。请务必将生成的文本严格放在 <START> 和 <END> 标签之间。任何其他消息或元数据都不允许出现在这些标签之外。

    请仔细阅读以下内容：
    严格遵循给定的小说大纲（包括字数要求和主要情节要点）。

    大纲：
    {outline}"""   
    ret_nstart = f"""目标：根据以下类型 {genres}，以{century}世纪为背景，用你自己的风格创作一部总字数为 {total_words} 字的小说。小说将按顺序分段生成。

    任务：继续小说的生成过程，生成小说的下一段内容（约 {NOVEL_SEGMENT_WORDS} 字）。在叙事中充分展开所有相关细节，并严格遵循大纲中规定的字数限制。不要匆忙推进任何情节节点；如果需要，可以将内容分成多个部分，以遵守字数限制。请务必将生成的文本严格放在 <START> 和 <END> 标签之间。如果小说已经完成，请在生成文本的末尾添加 "END OF NOVEL"。任何其他消息或元数据都不允许出现在这些标签之外。

    说明：
    1. 严格遵循小说大纲，包括字数要求和主要情节要点。
    2. 与上一段内容以及当前为止的整体摘要保持完全连贯。
    3. 使用之前的段落和摘要作为参考，以保持风格、语气和故事的一致性。

    大纲：
    {outline}

    上一段内容的结尾：
    {last_segment}

    截至目前的小说摘要：
    {summary}"""
    return ret_start if start else ret_nstart

def get_prompt_novel_ru(start, total_words, genres, NOVEL_SEGMENT_WORDS, outline, century, last_segment, summary):
    ret_start = f"""ЦЕЛЬ: Напиши роман объёмом {total_words} слов в собственном стиле на основе следующих жанров: {genres}. Действие романа происходит в {century} веке. Мы генерируем полный роман последовательно и по частям.

    ЗАДАЧА: Начни писать роман, сгенерировав начальный сегмент (~{NOVEL_SEGMENT_WORDS} слов). Развивай повествование всесторонне, включая все релевантные детали, строго придерживаясь ограничения по количеству слов, указанного в плане. Не спеши проходить через какие-либо ключевые этапы; при необходимости допустимо разделить содержание на несколько частей, чтобы соблюсти ограничения по количеству слов. Пожалуйста, заключи сгенерированный текст строго между тегами <START> и <END>. Любое другое сообщение или метаданные вне этих тегов.

    Пожалуйста, внимательно прочитай следующий текст:
    Строго следуй данному плану романа (включая количество слов и основные сюжетные точки).

    ПЛАН:
    {outline}"""
    ret_nstart = f"""ЦЕЛЬ: Напиши роман объёмом {total_words} слов в собственном стиле на основе следующих жанров: {genres}. Действие романа происходит в {century} веке. Мы генерируем полный роман последовательно и по частям.

    ЗАДАЧА: Ты продолжаешь процесс генерации романа. Сгенерируй следующий сегмент (~{NOVEL_SEGMENT_WORDS} слов) романа. Развивай повествование всесторонне, включая все релевантные детали, строго придерживаясь ограничения по количеству слов, указанного в плане. Не спеши проходить через какие-либо ключевые этапы; при необходимости допустимо разделить содержание на несколько частей, чтобы соблюсти ограничения по количеству слов. Пожалуйста, заключи сгенерированный текст строго между тегами <START> и <END>. Если роман завершён, добавь "END OF NOVEL" в конце сгенерированного текста. Любое другое сообщение или метаданные вне этих тегов.

    ИНСТРУКЦИИ:
    1. Строго следуй плану романа, включая количество слов и основные сюжетные точки.
    2. Сохраняй полную преемственность с предыдущим сегментом и общим кратким содержанием, сгенерированным до этого момента.
    3. Используй предыдущие сегменты и краткое содержание как ориентир для стиля, тона и последовательности истории.

    ПЛАН:
    {outline}

    ОКОНЧАНИЕ ПРЕДЫДУЩЕГО СЕГМЕНТА:
    {last_segment}

    КРАТКОЕ СОДЕРЖАНИЕ РОМАНА ДО ЭТОГО МОМЕНТА:
    {summary}"""
    return ret_start if start else ret_nstart

PROMPT_GEN_NOVEL = {
    "en": get_prompt_novel_en,
    "it": get_prompt_novel_it,
    "es": get_prompt_novel_es,
    "de": get_prompt_novel_de,
    "ch": get_prompt_novel_ch,
    "ru": get_prompt_novel_ru
}


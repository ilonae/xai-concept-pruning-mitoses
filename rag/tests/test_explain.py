from explain import heatmap_html, keyword_overlap, sentence_relevance

# sentence_relevance()
class TestSentenceRelevance:

    def test_returns_one_tuple_per_sentence(self, embed_model):
        text = (
            "CRP extends LRP to individual concepts. "
            "Each concept corresponds to a convolutional filter. "
            "Relevance is propagated layer by layer."
        )
        result = sentence_relevance("What is CRP?", text, embed_model)
        assert len(result) == 3

    def test_scores_are_floats_between_minus_one_and_one(self, embed_model):
        text = (
            "Concept pruning removes irrelevant filters. "
            "A binary mask is applied to the weights."
        )
        result = sentence_relevance("How does pruning work?", text, embed_model)
        for _, score in result:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0

    def test_most_relevant_sentence_scores_highest(self, embed_model):
        """
        The sentence that mentions the query topic should score higher than unrelated sentences
        """
        text = (
            "The weather today is sunny and warm. "
            "CRP attributes network decisions to individual convolutional concepts. "
            "Histopathology images show tissue at microscopic resolution."
        )
        result = sentence_relevance("concept relevance propagation attribution", text, embed_model)
        crp_score = next(s for sent, s in result if "CRP" in sent)
        weather_score = next(s for sent, s in result if "weather" in sent)
        assert crp_score > weather_score, (
            f"Expected CRP sentence to score higher than weather sentence, "
            f"got {crp_score:.3f} vs {weather_score:.3f}"
        )

    def test_short_sentences_are_filtered(self, embed_model):
        """Sentences shorter than 15 chars dropped"""
        text = "OK. CRP extends LRP to individual concept encoders in each layer."
        result = sentence_relevance("concept relevance", text, embed_model)
        sentences = [s for s, _ in result]
        assert not any(len(s) < 15 for s in sentences)

    def test_empty_like_input_returns_fallback(self, embed_model):
        """If no valid sentences are found, returns with score 1.0"""
        result = sentence_relevance("query", "Hi.", embed_model)
        assert len(result) == 1
        assert result[0][1] == 1.0

    def test_original_order_preserved(self, embed_model):
        sentences = [
            "First sentence about mitosis detection.",
            "Second sentence about concept attribution.",
            "Third sentence about layer-wise relevance.",
        ]
        text = " ".join(sentences)
        result = sentence_relevance("relevance", text, embed_model)
        returned_sentences = [s for s, _ in result]
        for orig, ret in zip(sentences, returned_sentences):
            assert orig.strip() == ret.strip()

# heatmap_html()

class TestHeatmapHtml:

    def _make_scores(self, pairs):
        return [(text, score) for text, score in pairs]

    def test_returns_string(self):
        scores = self._make_scores([("Hello world.", 0.5), ("CRP is great.", 0.9)])
        result = heatmap_html(scores)
        assert isinstance(result, str)

    def test_contains_all_sentences(self):
        scores = self._make_scores([
            ("First sentence here.", 0.3),
            ("Second sentence about CRP.", 0.8),
            ("Third sentence about LRP.", 0.5),
        ])
        result = heatmap_html(scores)
        assert "First sentence here." in result
        assert "Second sentence about CRP." in result
        assert "Third sentence about LRP." in result

    def test_uses_span_tags(self):
        scores = self._make_scores([("Some text.", 0.5), ("More text.", 0.9)])
        result = heatmap_html(scores)
        assert result.count("<span") == 2
        assert result.count("</span>") == 2

    def test_background_color_present(self):
        scores = self._make_scores([("Text A.", 0.2), ("Text B.", 0.9)])
        result = heatmap_html(scores)
        assert "background:" in result or "background-color:" in result

    def test_single_sentence_does_not_crash(self):
        scores = self._make_scores([("Only one sentence.", 0.7)])
        result = heatmap_html(scores)
        assert "Only one sentence." in result

# keyword_overlap()

class TestKeywordOverlap:

    def test_finds_shared_terms(self):
        query = "How does concept pruning work in neural networks?"
        chunk = "Concept pruning removes filters based on CRP attribution scores."
        result = keyword_overlap(query, chunk)
        assert "concept" in result
        assert "pruning" in result

    def test_stopwords_excluded(self):
        query = "What is the role of heatmaps in XAI?"
        chunk = "The heatmaps show the role of each concept in the decision."
        result = keyword_overlap(query, chunk)
        for stopword in ["what", "the", "of", "in", "is"]:
            assert stopword not in result

    def test_short_words_excluded(self):
        """Words shorter than min_len=4 should be filtered."""
        query = "CRP LRP XAI are all XAI methods"
        chunk = "CRP and LRP are XAI methods used for attribution"
        result = keyword_overlap(query, chunk, min_len=4)
        for word in result:
            assert len(word) >= 4

    def test_case_insensitive(self):
        query = "Concept Relevance Propagation"
        chunk = "concept relevance propagation extends lrp"
        result = keyword_overlap(query, chunk)
        assert "concept" in result
        assert "relevance" in result
        assert "propagation" in result

    def test_no_overlap_returns_empty(self):
        query = "quantum physics thermodynamics"
        chunk = "mitosis detection histopathology neural network"
        result = keyword_overlap(query, chunk)
        assert result == []

    def test_result_is_sorted(self):
        query = "pruning relevance concept attribution"
        chunk = "relevance attribution concept pruning filters"
        result = keyword_overlap(query, chunk)
        assert result == sorted(result)

    def test_german_stopwords_excluded(self):
        query = "Wie funktioniert das Konzept-Pruning?"
        chunk = "Das Konzept-Pruning entfernt irrelevante Filter durch Maskierung."
        result = keyword_overlap(query, chunk)
        for stopword in ["das", "wie", "durch"]:
            assert stopword not in result

    def test_returns_list(self):
        result = keyword_overlap("test query here", "test chunk text here")
        assert isinstance(result, list)

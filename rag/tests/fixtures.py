
FIXTURE_DOCUMENTS = [
    {
        "id": "doc_crp_1",
        "text": (
            "Concept Relevance Propagation (CRP) extends Layer-wise Relevance "
            "Propagation (LRP) by attributing network decisions not only to input "
            "pixels but to individual convolutional filters, called concepts. "
            "Each concept corresponds to a specific pattern the network has learned."
        ),
        "source": "crp_paper.pdf",
        "page": "1",
    },
    {
        "id": "doc_crp_2",
        "text": (
            "In CRP, relevance is propagated through the network layer by layer. "
            "At each layer, the top-k most relevant concept encoders are identified "
            "using a voting mechanism across training samples. "
            "This produces a concept atlas mapping each layer to its dominant concepts."
        ),
        "source": "crp_paper.pdf",
        "page": "2",
    },
    {
        "id": "doc_lrp_1",
        "text": (
            "Layer-wise Relevance Propagation (LRP) assigns a relevance score to "
            "each input feature by propagating the model output backwards through "
            "the network. The sum of all relevance scores equals the model output, "
            "satisfying a conservation property."
        ),
        "source": "lrp_paper.pdf",
        "page": "1",
    },
    {
        "id": "doc_pruning_1",
        "text": (
            "Concept pruning removes convolutional filters identified as irrelevant "
            "by their CRP attribution scores. A binary mask is applied to the filter "
            "weights, setting irrelevant concept activations to zero without "
            "retraining the full network."
        ),
        "source": "icw_report.pdf",
        "page": "5",
    },
    {
        "id": "doc_pruning_2",
        "text": (
            "The pruning strategy targets filters in the middle and higher layers, "
            "specifically features.22, features.25, and the classifier layers. "
            "Lower layers encode shared low-level texture features and are not pruned "
            "to avoid affecting other classes."
        ),
        "source": "icw_report.pdf",
        "page": "6",
    },
    {
        "id": "doc_dataset_1",
        "text": (
            "The dataset consists of histopathology images from the HTW Berlin CBMI "
            "mitosis detection benchmark. Images are 64x64 pixel patches labelled "
            "as Mitosis or non-Mitosis. The model is a fine-tuned VGG network "
            "trained on 40 samples per class."
        ),
        "source": "icw_report.pdf",
        "page": "3",
    },
    {
        "id": "doc_heatmap_1",
        "text": (
            "Heatmaps are generated for each concept by visualising the spatial "
            "distribution of relevance within the input image. High relevance regions "
            "indicate where the network focused when activating a specific concept. "
            "Composite heatmaps combine multiple concept visualisations into one view."
        ),
        "source": "icw_report.pdf",
        "page": "4",
    },
    {
        "id": "doc_tcav_1",
        "text": (
            "TCAV (Testing with Concept Activation Vectors) measures the sensitivity "
            "of a model's predictions to human-defined concepts. Unlike CRP, TCAV "
            "requires labelled concept examples and trains a linear classifier "
            "in the model's activation space."
        ),
        "source": "tcav_paper.pdf",
        "page": "1",
    },
]

EVAL_QUERIES = [
    {
        "query": "What is Concept Relevance Propagation?",
        "expected_sources": ["crp_paper.pdf"],
        "expected_keywords": ["crp", "concept", "relevance", "propagation"],
    },
    {
        "query": "How does concept pruning work?",
        "expected_sources": ["icw_report.pdf"],
        "expected_keywords": ["prun", "mask", "filter", "concept"],
    },
    {
        "query": "What dataset was used?",
        "expected_sources": ["icw_report.pdf"],
        "expected_keywords": ["mitosis", "htw", "dataset", "image"],
    },
    {
        "query": "What is the role of heatmaps in XAI?",
        "expected_sources": ["icw_report.pdf"],
        "expected_keywords": ["heatmap", "visual", "relevance", "concept"],
    },
    {
        "query": "How does CRP differ from LRP?",
        "expected_sources": ["crp_paper.pdf", "lrp_paper.pdf"],
        "expected_keywords": ["crp", "lrp", "concept", "relevance"],
    },
]

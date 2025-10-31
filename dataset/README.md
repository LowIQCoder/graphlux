# Dataset Information

Here you can find information about our dataset preparation. We decided to construct our own dataset using information from open sources such as books and courses. Estimated size of final dataset is ~800 unique nodes

## Node Example
All nodes listed in our final [dataset](nodes.json) have the same structure

```json
{
    "name": "<name of topic>",
    "tags": ["<tag_1>", "<tag_2>"],
    "summary": "<sumarry of extracted topic>",
    "content": "<extracted contend>",
    "source": "book"
}
```

## Scrapping Progress
- [x] [Think Python](https://allendowney.github.io/ThinkPython/)
    - 171 unique nodes
    - see preprocessing [code](./think_python/process.py)
- [ ] [PyFlo](https://pyflo.net/)
- [ ] [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/#toc)
- [ ] [Natural Language Processing with Python](https://www.nltk.org/book/)
- [ ] [Programming Computer Vision with Python](http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf)
- [ ] [The Hitchhiker’s Guide to Python!](https://docs.python-guide.org/)

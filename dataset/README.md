# Dataset Information

Here you can find information about our dataset preparation. We decided to construct our own dataset using information from open sources such as books and courses

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
- [x] https://allendowney.github.io/ThinkPython/
    - 171 unique nodes
    - see preprocessing [code](./think_python/process.py)
- [ ] https://pyflo.net/
- [ ] https://automatetheboringstuff.com/#toc
- [ ] https://www.nltk.org/book/
- [ ] http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf
- [ ] https://docs.python-guide.org/

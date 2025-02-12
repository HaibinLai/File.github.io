# HPC Lab 

build with mkdocs

The website is building in the following commands.

1. Install mkdocs and material theme that support the website
```bash
pip install mkdocs mkdocs-material
```

2. modify the website blogs in `/docs`. write markdown blogs.

If we want to add new page to the website, go to /mkdocs.yml and set up `nav` part 

3. building the website

Now we can compile the markdown into html formula. Use the following commman:
```bash
mkdocs build
```

This will form a `/site` folder that display the site.

we can check our website's appearance:
```bash
mkdocs serve
```

4. Uploading and deploy on github

Just use the following command:

```bash
mkdocs gh-deploy --clean
```


## More info

https://blog.csdn.net/qq_41261251/article/details/116021097

https://www.cnblogs.com/chinjinyu/p/17610438.html

https://squidfunk.github.io/mkdocs-material/

https://www.mkdocs.org/
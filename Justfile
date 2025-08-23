
default:
    @just --list

# commit and push everything so runpod can pull
push:
    git add . && git commit -m "pushing" && git push

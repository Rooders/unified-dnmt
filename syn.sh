#!/bin/sh
git status  
git add *  
git commit -m 'add some code from Mac'
# git commit -m 'add some results from Server'
git push origin main            #upload data
git stash pop


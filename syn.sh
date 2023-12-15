#!/bin/sh
git status  
git add *  
git commit -m 'add some code from Mac'
git commit -m 'add some results from Server'
git pull --rebase origin main   #domnload data
git push origin main            #upload data
git stash pop



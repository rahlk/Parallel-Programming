typo: ready
	@- env GIT_SSL_NO_VERIFY=true git add --all .	
	@- env GIT_SSL_NO_VERIFY=true git status
	@- env GIT_SSL_NO_VERIFY=true git commit -am "commit with a makefile"
	@- env GIT_SSL_NO_VERIFY=true git "SSL_NO_VERIFY=TRUE" push origin master 

commit: ready 
	@- git status
	@- git commit -a
	@- git push origin master

update: ready
	@- git pull origin master

status: ready
	@- git status

ready: ready
	@git config --global credential.helper cache
	@git config credential.helper 'cache --timeout=3600'

rahlk:  # <== change to your name
	@git config --global user.name "rahlk"
	@git config --global user.email i.m.ralk@gmail.com

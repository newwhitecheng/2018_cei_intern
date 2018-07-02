## VNC set up
1. Download VNC viewer and install it

2. Use Xshell to connect to the server, with IP as 10.236.176.23

3. Run 
 ```Shell 
    vncserver -geometry 2560x1440
 ```
to open a new prot of your own. (“2560x1440” is the resolution of your monitor, 'x' is the lowecase letter 'x')

4. Open VNC on you computer, enter IP and port number as 10.236.176.23:* (* is your port number)

5. To kill you port, run
```Shell 
    vncserver -kill *
 ```

6. Do not kill the port of other interns!

## data server set up
1. Make your local mount point. For example ‘mkdir ~/share’. This only needs to be done once.
2. Mount the fileserver using command ‘sudo mount -o username='WIN\[YourDukeNetID]',password='[YourDukeNetIDPassword]',uid=’[YourLinuxLocalUsername]’,gid=’[YourLinuxLocalGroupname]’ //10.148.54.21/Share  [mount point]’. Make sure to replace all the [xxx] placeholders accordingly and don’t remove the single quote marks.
3. Now you should be able to use it just as a local disk.
Note that it is a good security practice that you clear your command history afterwards by ‘history -c’, since your DukeNetID and its password is stored in your command history as plain text after step 2.

# Label review

For each item, open the PDF at the page shown (the page number in the PDF viewer's toolbar, not the number printed on the page) and check: the evidence answers the question; the key facts are what a correct answer must say; nothing else in the manual answers it better (if it does, it must be added as evidence). Unanswerable: search the PDF yourself (Ctrl+F) for the terms listed, plus any other wording you can think of.


## direct

- [ ] **h-d-tp-01**: How do I set up my TP-Link Archer AX55 as an access point?
  - brand filter expected: tplink | answerable: True
  - evidence: `tplink_archer_ax55.pdf` pp. 22-22, key phrase "Set Up the Router as an Access Point"  
    p. 22: …4 4 Set Up the Router as an Access Point The router can work as an access point, transforming your existing wired network to a wireless one. 1. Visit http://tplinkwifi.net, and log in with your TP…
  - key facts: Operation Mode; Access Point; Quick Setup
- [ ] **h-d-dl-01**: How do I create a schedule on my D-Link router?
  - brand filter expected: dlink | answerable: True
  - evidence: `dlink_dir_x1560.pdf` pp. 86-86, key phrase "Add a Schedule"  
    p. 86: …pencil icon in the Edit column. If you wish to create a new schedule, click the Add a Schedule button.…
  - key facts: Time & Schedule; Add a Schedule
- [ ] **h-d-as-01**: How do I set up DDNS on my ASUS router?
  - brand filter expected: asus | answerable: True
  - evidence: `asus_rt_ax58u.pdf` pp. 76-77, key phrase "To set up DDNS"  
    p. 76: …de your network through the provided ASUS DDNS Service or another DDNS service. To set up DDNS: 1. From the navigation panel, go to Advanced Settings > WAN > DDNS. 2. Configu…
  - key facts: WAN > DDNS

## paraphrased

- [ ] **h-p-tp-01**: How can I limit how much screen time my kids get with the TP-Link router?
  - brand filter expected: tplink | answerable: True
  - evidence: `tplink_archer_ax55.pdf` pp. 52-53, key phrase "Manage screen time"  
    p. 52: …ection Keep your child away from inappropriate content Family Incentive Program Manage screen time and create rewards Family Time Pause the internet to enjoy family time - To use…
  - key facts: Tether; HomeShield
  - note: The manual covers this only through the Tether app (HomeShield), not the web page.
- [ ] **h-p-as-01**: A firmware update on my ASUS router went wrong and now the power light just flashes slowly. How do I fix it?
  - brand filter expected: asus | answerable: True
  - evidence: `asus_rt_ax58u.pdf` pp. 92-93, key phrase "Firmware Restoration is used"  
    p. 92: …4.2 Firmware Restoration Firmware Restoration is used on an ASUS Wireless Router that failed during its firmware upgrading process. It uploads the firmware that you specify. The process ta…
  - key facts: rescue mode; Firmware Restoration
- [ ] **h-p-te-01**: I'd like my Tenda router to restart by itself every night so it stays stable. Is that possible?
  - brand filter expected: tenda | answerable: True
  - evidence: `tenda_wifi6.pdf` pp. 150-150, key phrase "Auto system maintenance enables you to restart"  
    p. 150: …9.11.5 Auto system maintenance Auto system maintenance enables you to restart the router regularly. It helps improve the stability and service life of the router. To access the configuration page, log in to…
  - key facts: Auto System Maintenance; Reboot At
- [ ] **h-p-ng-01**: My home IP address keeps changing. Can I reach my Netgear router from outside using a name instead?
  - brand filter expected: netgear | answerable: True
  - evidence: `netgear_rax50.pdf` pp. 120-121, key phrase "Set up a new Dynamic DNS account"  
    p. 120: …service does not work because private addresses are not routed on the Internet. Set up a new Dynamic DNS account To set up Dynamic DNS and register for a free NETGEAR account: 1. Launch a web …
  - key facts: Dynamic DNS; mynetgear.com

## no_brand

- [ ] **h-nb-01**: How do I change the router's LAN IP address?
  - brand filter expected: none | answerable: True
  - evidence: `tplink_archer_ax55.pdf` pp. 82-82, key phrase "Change the LAN Settings"  
    p. 82: …14 1 Change the LAN Settings The router is preset with a default LAN IP 192.168.0.1, which you can use to log in to its web management page. The LAN IP address together with the Subne…
  - evidence: `netgear_rax50.pdf` pp. 58-59, key phrase "To change the LAN TCP/IP settings"  
    p. 59: …es on the network use, or if you use competing subnets with the same IP scheme. To change the LAN TCP/IP settings: 1. Launch a web browser from a computer or mobile device that is connected to …
  - evidence: `asus_rt_ax58u.pdf` pp. 48-48, key phrase "To modify the LAN IP settings"  
    p. 48: …3.9 LAN 3.9.1 LAN IP The LAN IP screen allows you to modify the LAN IP settings of your wireless router. NOTE: Any changes to the LAN IP address will be reflected on your DHCP settings. To …
  - evidence: `dlink_dir_x1560.pdf` pp. 67-67, key phrase "LAN IP Address Enter the IP address of the router"  
    p. 67: …ve at any time to save the changes you have made on this page. Network Settings LAN IP Address Enter the IP address of the router. The default IP address is 192.168.0.1. If you change the IP address, once you …
  - evidence: `tenda_wifi6.pdf` pp. 102-102, key phrase "Network Settings > LAN Settings"  
    p. 102: …s the configuration page, log in to the web UI of the router, and choose More > Network Settings > LAN Settings. Overview On this page, you can: ◼ Change the LAN IP address and subnet mask of…

## tables

- [ ] **h-t-ng-01**: How much does the Netgear RAX50 weigh?
  - brand filter expected: netgear | answerable: True
  - evidence: `netgear_rax50.pdf` pp. 159-159, key phrase "1.25 lbs (566 g)"  
    p. 159: …C output Power adapter 11.65 x 8.11 x 2.25 in (295.8 x 206 x 57.2 mm)Dimensions 1.25 lbs (566 g)Weight 0° to 40°C (32o to 104oF)Operating temperature 90% maximum relative humid…
  - key facts: 566 g | 1.25 lb
- [ ] **h-t-te-01**: What does it mean when the LED on my Tenda router blinks yellow quickly?
  - brand filter expected: tenda | answerable: True
  - evidence: `tenda_wifi6.pdf` pp. 11-11, key phrase "Batch upgrade succeeded"  
    p. 11: …iled Reset Blinking red quickly Resetting Batch upgrade Blinking yellow quickly Batch upgrade succeeded Solid yellow Batch upgrade failed…
  - key facts: Batch upgrade succeeded
  - must not contain: Batch upgrade failed
  - note: Solid yellow in the same table means the batch upgrade failed.

## context_confusion

- [ ] **h-c-ng-01**: How do I open the files on a USB drive plugged into my Netgear router from a Windows PC?
  - brand filter expected: netgear | answerable: True
  - evidence: `netgear_rax50.pdf` pp. 113-113, key phrase "router from a Windows-based computer"  
    p. 113: …Access a storage device connected to the router from a Windows-based computer To access the USB storage device from a Windows-based computer: 1. Connect a USB storage device to a USB port on your …
  - key facts: readyshare; Start > Run | Run dialog
  - must not contain: smb://readyshare; Connect to Server
  - note: The Mac procedure (smb://readyshare) is on the next page.
- [ ] **h-c-tp-01**: How do I set up a PPTP VPN server on my TP-Link router?
  - brand filter expected: tplink | answerable: True
  - evidence: `tplink_archer_ax55.pdf` pp. 75-76, key phrase "Use PPTP VPN to Access Your Home Network"  
    p. 75: …alled. 3. Run the OpenVPN client utility and connect it to OpenVPN Server. 13 2 Use PPTP VPN to Access Your Home Network PPTP VPN Server is used to create a VPN connection for remote device. To use th…
  - key facts: VPN Server > PPTP; Client IP Address
  - must not contain: certificate; EXPORT; Service Type
  - note: The OpenVPN server steps (certificate, EXPORT) are on the same page.

## unanswerable

- [ ] **h-u-ng-01**: How do I add an Orbi mesh satellite to my Netgear Nighthawk RAX50?
  - brand filter expected: netgear | answerable: False
  - absence: searched ['Orbi', 'mesh', 'satellite'] in ['netgear_rax50']
- [ ] **h-u-te-01**: How do I set up a WireGuard VPN on my Tenda router?
  - brand filter expected: tenda | answerable: False
  - absence: searched ['WireGuard', 'OpenVPN'] in ['tenda_ax3000']

## out_of_scope

- [ ] **h-o-01**: Can you recommend a good laptop for programming?
  - brand filter expected: none | answerable: False

## conversations

## Absence checks (automatic)

- h-u-ng-01: 'Orbi', 'mesh', 'satellite' in netgear_rax50 (raw + cleaned): 0 hits
- h-u-te-01: 'WireGuard', 'OpenVPN' in tenda_ax3000 (raw + cleaned): 0 hits

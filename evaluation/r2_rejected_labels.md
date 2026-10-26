# R2: rejected section labels (rag_500_60_ctx, header version 1.1)

Section labels the trusted-label rule did not use, and what the header used instead:
- a trusted heading on the **same page** (fallback), or
- `(none)`: the header is brand and model only.

Sections: trusted 332, fallback 2, omitted 98 (of 432 chunks). Headers shortened to fit 512 tokens: {'no_section': 11, 'brand_only': 6}.

Version 1.0 allowed a fallback from up to 2 pages back. Three of those were wrong topics (D-Link "Time Configuration" → "Quick VPN", Netgear MU-MIMO → "Delete a static route", Tenda "PPTP server" → "DNS"), so 1.1 allows the same page only.

| Manual | Rejected label | Header used instead | Chunks |
|---|---|---|---|
| asus_rt_ax58u | Terms & conditions for copying, distribution, & modification | (none) | 6 |
| asus_rt_ax58u | Two-Way IPS | (none) | 1 |
| asus_rt_ax58u | Using the FTP Share service | (none) | 1 |
| asus_rt_ax58u | PPTP Passthrough, L2TP Passthrough, IPsec Passthrough and RTSP | (none) | 1 |
| asus_rt_ax58u | 4 Utilities4 Utilities | (none) | 1 |
| asus_rt_ax58u | Administration > Restore/Save/Upload Setting and click | (none) | 1 |
| asus_rt_ax58u | Windows® | (none) | 1 |
| asus_rt_ax58u | Licensing information | (none) | 1 |
| asus_rt_ax58u | AppendicesAppendices | (none) | 1 |
| dlink_dir_x1560 | Lifetime | (none) | 3 |
| dlink_dir_x1560 | Modem | (none) | 2 |
| dlink_dir_x1560 | 2.4GHz / 5GHz | (none) | 2 |
| dlink_dir_x1560 | Déclaration d'exposition aux radiations | (none) | 2 |
| dlink_dir_x1560 | ErP Power Usage | (none) | 1 |
| dlink_dir_x1560 | Always Up-to-Date with the Latest Features | (none) | 1 |
| dlink_dir_x1560 | Address | (none) | 1 |
| dlink_dir_x1560 | DHCP-PD in LAN | (none) | 1 |
| dlink_dir_x1560 | IPv6 DNS Settings | (none) | 1 |
| dlink_dir_x1560 | Range (Start) | (none) | 1 |
| dlink_dir_x1560 | Range | (none) | 1 |
| dlink_dir_x1560 | RTSP Allows applications that uses Real Time Streaming Protocol | (none) | 1 |
| dlink_dir_x1560 | Time Configuration | (none) | 1 |
| dlink_dir_x1560 | Notification | (none) | 1 |
| dlink_dir_x1560 | LED Control | (none) | 1 |
| dlink_dir_x1560 | Click Start > Settings > Network & Internet > | (none) | 1 |
| dlink_dir_x1560 | Section 7 - Troubleshooting | (none) | 1 |
| dlink_dir_x1560 | Wireless Local Area Network (WLAN) | (none) | 1 |
| dlink_dir_x1560 | Small Office and Home Office Uses/Benefits | (none) | 1 |
| dlink_dir_x1560 | Appendix B - Wireless Modes | (none) | 1 |
| dlink_dir_x1560 | Statically Assign an IP address | (none) | 1 |
| dlink_dir_x1560 | A-PSK (Pre-Shared Key) | (none) | 1 |
| dlink_dir_x1560 | Device Interfaces | (none) | 1 |
| dlink_dir_x1560 | FCC Radiation Exposure Statement | (none) | 1 |
| dlink_dir_x1560 | EIN DRAHTLOSES PRODUKT ) | (none) | 1 |
| dlink_dir_x1560 | POUR LES PRODUITS SANS FIL) | (none) | 1 |
| dlink_dir_x1560 | INALÁMBRICO) | (none) | 1 |
| dlink_dir_x1560 | SAFETY INSTRUCTIONS | (none) | 1 |
| dlink_dir_x1560 | SICHERHEITSVORSCHRIFTEN | (none) | 1 |
| dlink_dir_x1560 | D-Link und die Umwelt | (none) | 1 |
| dlink_dir_x1560 | Appendix H - Wall Mounting Information | (none) | 1 |
| netgear_rax50 | Support and Community | (none) | 1 |
| netgear_rax50 | Figure 7. Cable your router | (none) | 1 |
| netgear_rax50 | Manually set up the Internet connection on page 24.) | Set up an IPv6 fixed Internet connection | 1 |
| netgear_rax50 | Multiuser multiple input, multiple output (MU-MIMO) improves performan | (none) | 1 |
| netgear_rax50 | Reply from < IP address >: bytes=32 time=NN ms TTL=xxx | (none) | 1 |
| netgear_rax50 | Maximum WiFi signal rate complies with the IEEE® 802.11 standard.2 | (none) | 1 |
| tenda_ax3000 | Parameter Description | (none) | 3 |
| tenda_ax3000 | Assigned IP | (none) | 2 |
| tenda_ax3000 | An example of enabling internet users to access LAN resources | (none) | 2 |
| tenda_ax3000 | Copyright Statement | (none) | 1 |
| tenda_ax3000 | Type | (none) | 1 |
| tenda_ax3000 | Digi: Digi-TM, Digi, Digi-CT Sabah and Digi-TNB | (none) | 1 |
| tenda_ax3000 | MAC Address Clone | (none) | 1 |
| tenda_ax3000 | Scenario Connection Type | (none) | 1 |
| tenda_ax3000 | PPTP server | (none) | 1 |
| tenda_ax3000 | Watch IPTV programs through the router | (none) | 1 |
| tenda_ax3000 | 100 Mbps half duplex | (none) | 1 |
| tenda_ax3000 | Only prohibit specified clients from accessing the internet | (none) | 1 |
| tenda_ax3000 | Local upgrade | (none) | 1 |
| tplink_archer_ax55 | Home Network Router Remote Devices | (none) | 2 |
| tplink_archer_ax55 | More Info | About This Guide | 1 |
| tplink_archer_ax55 | WPS/Wi-Fi Button | (none) | 1 |
| tplink_archer_ax55 | OR Smart DeviceComputer | (none) | 1 |
| tplink_archer_ax55 | Protects your home network from DDoS attacks | (none) | 1 |
| tplink_archer_ax55 | Personal Website Router | (none) | 1 |
| tplink_archer_ax55 | FCC RF Radiation Exposure Statement | (none) | 1 |
| tplink_archer_ax55 | Model Number: FCC ID: 2AXJ4AX55 GQ24-120200-AU, FCC ID: 2AXJ4AX55V2 T1 | (none) | 1 |
| tplink_archer_ax55 | CE Mark Warning | (none) | 1 |
| tplink_archer_ax55 | Storage Temperature:-40°C~60°C | (none) | 1 |

var system = require('system');
var args = system.args;

var fs = require('fs');
var page = require('webpage').create();
page.content = fs.read(args[1]);
page.render(args[2], {format: 'jpeg', quality: '100'});
phantom.exit();
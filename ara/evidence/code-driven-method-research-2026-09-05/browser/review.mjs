import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [profile, report, destination] = process.argv.slice(2);
const port = (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0];
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {socket.onopen=resolve; socket.onerror=reject;});
let sequence=0;
const pending=new Map(), listeners=new Set();
socket.onmessage=event => {
  const message=JSON.parse(event.data);
  if (message.id) {
    const job=pending.get(message.id);
    pending.delete(message.id);
    if(message.error) job.reject(new Error(JSON.stringify(message.error)));
    else job.resolve(message.result);
  } else for(const listener of listeners) listener(message);
};
function command(method, params={}) {
  return new Promise((resolve,reject) => {
    const id=++sequence;
    pending.set(id,{resolve,reject});
    socket.send(JSON.stringify({id,method,params}));
  });
}
function loaded() {
  return new Promise((resolve,reject) => {
    const timeout=setTimeout(() => {listeners.delete(listener);reject(new Error('Page load timeout'));},60000);
    const listener=message => {if(message.method==='Page.loadEventFired') {
      clearTimeout(timeout);listeners.delete(listener);resolve();
    }};
    listeners.add(listener);
  });
}
async function evaluate(expression) {
  const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function navigate(url) {
  const ready=loaded();
  const result=await command('Page.navigate',{url});
  if(result.errorText) throw new Error(result.errorText);
  await ready;
}
async function click(selector) {
  const ready=loaded();
  await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  await ready;
}
await command('Page.enable');
await command('Runtime.enable');
await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
const url=pathToFileURL(path.resolve(report)).href;
const label=path.basename(path.dirname(report));
await navigate(url);
const overview=await evaluate(`({title:document.title,rows:document.querySelectorAll('table tr').length-1,sections:document.querySelectorAll('section').length,images:document.images.length,broken:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),links:document.querySelectorAll('a').length})`);
if(overview.broken.length) throw new Error(JSON.stringify(overview.broken));
let shot=await command('Page.captureScreenshot',{format:'png'});
await fs.writeFile(path.join(destination,label+'-overview.png'),Buffer.from(shot.data,'base64'));
await click('a[href$="/reconstruction.png"]');
const native=await evaluate(`({url:location.href,width:document.images[0]?.naturalWidth,height:document.images[0]?.naturalHeight})`);
if(!native.width) throw new Error('Native image failed');
shot=await command('Page.captureScreenshot',{format:'png'});
await fs.writeFile(path.join(destination,label+'-native.png'),Buffer.from(shot.data,'base64'));
await navigate(url);
await click('a[href="metrics.json"]');
const raw=await evaluate(`({url:location.href,length:document.body.innerText.length,prefix:document.body.innerText.slice(0,120)})`);
if(!raw.url.endsWith('/metrics.json')||raw.length<100) throw new Error('Raw metric link failed');
const receipt={report,overview,native,raw};
await fs.writeFile(path.join(destination,label+'-browser.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
await command('Page.close');
socket.close();

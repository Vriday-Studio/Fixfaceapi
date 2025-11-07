const CONNECTION_POOL_MS = 5_000;
const MSG_TYPE = {
  Unknown: 0,
  Heartbeat: 1,
  PeopleData: 2,
  AudioData: 3,
  SessionStart: 4,
  SessionEnd: 5,
  Standby: 6,
  VocStart: 7,
  VocEnd: 8,
  CrowdStat: 9,
  MicData: 10,
};

const PolicyTypeEnum = {
  Empty: 0,
  Greet: 1,
  OnGreenZone: 2,
  RedZone: 3,
  SpeakerFocus: 4,
  AskGroupChange: 5,
  LeftZone: 6,
  CallOver: 7,
};

const anotherHandle = {
  emptHandle:0,
}

export default class NetConnectionUtils {
  websocket = null;

  constructor(websock, str_url) {
    this.websocket = websock;
    this.url = str_url;
  }

  connect() {
    this.websocket = new WebSocket(this.url);
    this.websocket.onopen = () => {
      console.log("WebSocket connected");
    };
    this.websocket.onclose = () => {
      console.log("WebSocket disconnected");
    };
  }

  send(data) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(data));
    } else {
      console.error("WebSocket is not open");
    }
  }

  close() {
    if (this.websocket) {
      this.websocket.close();
    }
  }
}

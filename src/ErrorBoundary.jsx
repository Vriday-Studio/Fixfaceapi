import React, { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props){ super(props); this.state = { error: null }; }
  static getDerivedStateFromError(err){ return { error: err }; }
  render(){
    if (this.state.error) return <div style={{padding:12}}>Something went wrong.</div>;
    return this.props.children;
  }
}
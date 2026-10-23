import { useEffect, useState } from "react";
import { getExplain } from "../api.js";

// Fetches the plain-language explanations once. Shows a clear message if the server is down.
export default function useExplain() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    getExplain()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, []);
  return { data, error };
}

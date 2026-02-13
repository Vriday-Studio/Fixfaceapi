export function buildVisitContext({
  clockRef,
  locationRef,
  weatherRef,
  totalsRef,
  extra = {},
}) {
  const now = clockRef?.current || new Date();
  return {
    iso: now.toISOString(),
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    dayName: now.toLocaleDateString("id-ID", { weekday: "long" }),
    dateLabel: now.toLocaleDateString("id-ID", {
      day: "numeric",
      month: "long",
      year: "numeric",
    }),
    timeLabel: now.toLocaleTimeString("id-ID", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    }),
    time24: now.toLocaleTimeString("id-ID", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }),
    location: locationRef?.current,
    weather: weatherRef?.current,
    totals: totalsRef?.current,
    ...extra,
  };
}

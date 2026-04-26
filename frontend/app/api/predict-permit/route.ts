const BACKEND_URL = process.env.PERMIT_PILOT_URL ?? 'http://localhost:5000';

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Invalid JSON in request body' }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/predict-permit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('content-type') ?? 'application/json' },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return Response.json(
      { error: `Could not reach permit_pilot at ${BACKEND_URL}: ${message}` },
      { status: 502 },
    );
  }
}

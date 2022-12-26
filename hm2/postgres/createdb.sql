CREATE TABLE public.dataset (
  "col0"           NUMERIC(18,4),
  "col1"           NUMERIC(18,4),
  "target"     INT
);

COPY public.dataset(
  "col0",
  "col1",
  "target"
) FROM '/var/lib/postgresql/data/data.csv' DELIMITER ',' CSV HEADER;


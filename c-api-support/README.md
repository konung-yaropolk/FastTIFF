# c-api-support

The scaffolding of the C APIs in [FastTIFF](https://github.com/konung-yaropolk/FastTIFF):
`fast-tiff-lib`, `oir-reader`, `suite2p-registration` and `turboreg-registration`.
One definition of what each used to carry a copy of:

- `status`: the statuses all four share by number (`OK` 0, `CANCELLED` 1,
  `INVALID_ARGUMENT` 2, `UNAVAILABLE` 3, `FAILED` 4); each library numbers its
  own from 5.
- `Errors`: a library's failure reporting around its own thread-local slot:
  `fail`, `last_error`, panic guards, `text_arg` for strings coming in, and
  `put_text` for strings going out the `snprintf` way, cut between characters
  so C never gets invalid UTF-8.

No dependencies. It exists for those crates; its API follows what they need.
Licensed MPL-2.0.

#from pathlib import Path
##from kedro.framework.session import KedroSession
#from kedro.framework.startup import bootstrap_project
#
#class TestKedroRun:
#    def test_kedro_run(self):
#        bootstrap_project(Path.cwd())
#
#        with KedroSession.create(project_path=Path.cwd(), env="test") as session:
#            assert session.run() is not None